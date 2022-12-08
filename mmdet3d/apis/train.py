# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg
from torch import distributed as dist

from mmdet3d.datasets import build_dataset
from mmdet3d.utils import find_latest_checkpoint
from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import get_root_logger as get_mmdet_root_logger
from mmseg.core import DistEvalHook as MMSEG_DistEvalHook
from mmseg.core import EvalHook as MMSEG_EvalHook
from mmseg.datasets import build_dataloader as build_mmseg_dataloader
from mmseg.utils import get_root_logger as get_mmseg_root_logger


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, optional): The seed. Default to None.
        device (str, optional): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    # rank：当前进程的排名（只能选择0或-1）
    # world_size：当前工作的进程数
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        # 将seed转换为tensor放入cuda
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    # .item()将tensor转换为int32，普通python类型，不再是tensor
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    此函数会对 python、numpy、torch 都设置随机数种子
    保持随机数种子相同时，卷积的结果在CPU上相同，在GPU上仍然不相同。这是因为，cudnn卷积行为的不确定性。
    使用 torch.backends.cudnn.deterministic = True 可以解决。
    cuDNN 使用非确定性算法，并且可以使用 torch.backends.cudnn.enabled = False 来进行禁用
    如果设置为 torch.backends.cudnn.enabled = True，说明设置为使用非确定性算法
    （即会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题）。
    一般来讲，应该遵循以下准则：
    1. 如果网络的输入数据维度或类型上变化不大，设置 torch.backends.cudnn.benchmark = true 可以增加运行效率
    2. 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    设置 torch.backends.cudnn.benchmark = False 避免重复搜索

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # manual_seed_all 是为所有 GPU 都设置随机数种子
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_mmseg_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_mmseg_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmseg_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = MMSEG_DistEvalHook if distributed else MMSEG_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    """
    train_detector()函数主要是构建了dataloader，初始化了优化器以及runner和hooks，最后调用runner.run开始正式的迭代训练流程
    在下面的train_model中调用此函数

    Args:
        model:
        dataset:
        cfg:
        distributed:
        validate:
        timestamp:
        meta:
    """
    # 获取 logger
    logger = get_mmdet_root_logger(log_level=cfg.log_level)

    # ------------------------prepare data loaders--------------------------
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # 获得 samples_per_gpu
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # 设置迭代性质：默认为EpochBased，若配置文件中有runner项，则按照配置文件判断是EpochBased或IterBased
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']

    # ------------Step1: 获取dataloader, 因为dataset列表里包含了训练集和验证集, 所以使用for循环的方式构建dataloader---------------
    # 加载dataset为PyTorch DataLoader形式
    data_loaders = [
        build_mmdet_dataloader(
            ds,  # dataset
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))  # return PyTorch DataLoader
        for ds in dataset  # dataset列表里包含训练集和验证集使用for循环构建dataloader
    ]

    # -------------------------Step2: 封装模型，进行分布式训练------------------------------
    # DP:单机多卡  DDP：单机多卡或多机多卡
    # 多卡训练
    # broadcast_buffers：在每次调用forward之前是否进行buffer的同步，比如bn中的mean和var，如果你实现了自己的SyncBn可以设置为False
    # find_unused_parameters：是否查找模型中未参与loss“生成”的参数，简单说就是模型中定义了一些参数但是没用上时需要设置
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    # 单卡训练
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # -------------------------Step3: 初始化优化器--------------------------
    optimizer = build_optimizer(model, cfg.optimizer)

    # -----------------Step4: 初始化Runner--------------------
    # 若配置文件runner不存在，则按照默认值EpochBased，total_epochs运行
    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    # ----------------------------build runner----------------------------
    runner = build_runner(
        cfg.runner,  # dict(type='EpochBasedRunner', max_epochs=xx)
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # -----------------------------fp16 setting------------------------------
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        # 使用半精度训练时，用Fp16OptimizerHook重新构建优化器hook
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # ---------------------------Step5: 注册默认Hook和自定义Hook(注册到runner._hooks列表中)-----------------------------
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    # 多卡训练钩子注册
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # ----------------------------register eval hooks------------------------------
    # 判断验证时batch_size，并创建验证数据流和eval_hook，注册eval_hook(val_dataloader, **eval_cfg)至runner
    if validate:  # bool
        # Support batch_size > 1 in validation
        # dict.pop(key,number):若存在key，则删除键值对并返回值；否则返回number
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)  # val中没有samples_per_gpu,故取1
        # 验证batch_size>1
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        # 创建验证数据集配置
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # 从data_loaders中创建验证数据流
        val_dataloader = build_mmdet_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,  # 1
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        # eval_hook
        eval_cfg = cfg.get('evaluation', {})  # evaluation = dict(interval=xx) hook
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = MMDET_DistEvalHook if distributed else MMDET_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')


    # ----------------------------------Step6: 开始训练流程------------------------------------
    resume_from = None  # 创建变量并初始化，避免bug
    # 判断启用手动续训或自动续训
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)  # latest.pth
    # 启用自动续训，从latest.pth
    if resume_from is not None:
        cfg.resume_from = resume_from
    # 启用手动续训，指定续训pth
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    # 迁移学习，从外部加载pth
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    # ----------------------------------runner.run----------------------------------
    runner.run(data_loaders, cfg.workflow)  # 调用run()方法, 开始迭代过程


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        # 构建dataloader，初始化优化器以及runner和hooks
        # 最后调用runner.run开始正式的迭代训练流程
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
