# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
# Config 用于读取配置文件, DictAction 将命令行字典类型参数转化为 key-value 形式
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', default='', help='train config file path')  # 训练配置文件地址
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    # action: store (默认, 表示保存参数)
    # action: store_true, store_false (如果指定参数, 则为 True, False)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    # -------------创建一个互斥组. argparse 将会确保互斥组中的参数只能出现一个--------------
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    # 可以使用 python train.py --gpu-ids 0 1 2 3 指定使用的 GPU id
    # 参数结果：[0, 1, 2, 3]
    # nargs = '*'：参数个数可以设置0个或n个
    # nargs = '+'：参数个数可以设置1个或n个
    # nargs = '?'：参数个数可以设置0个或1个
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    # ---------------------------------------------------------------------------

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # 其他参数: 可以使用 --options a=1,2,3 指定其他参数
    # 参数结果: {'a': [1, 2, 3]}
    parser.add_argument(
        '--options',  # 用键值对的形式修改配置文件
        nargs='+',  # 表示参数可以设一个或多个
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',  # 用键值对的形式修改配置文件，其中值可以是list或嵌套的list/tuple
        nargs='+',  # 表示参数可以设一个或多个
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    # 如果使用 dist_utils.sh 进行分布式训练, launcher 默认为 pytorch
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # 本地进程编号，此参数 torch.distributed.launch 会自动传入
    parser.add_argument('--local_rank', type=int, default=0)
    # 根据gpu数量自动裁剪lr
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    # 如果环境中没有 LOCAL_RANK，就设置它为当前的 local_rank
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    # ------------------------------1.从命令行和配置文件获取配置参数----------------------------------
    # 加载参数
    args = parse_args()
    # 加载训练配置文件  args.config是path
    cfg = Config.fromfile(args.config)
    # 在训练终端窗口读取传入的新配置参数
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark，设置True 可以加速输入大小固定的模型训练和推理
    # 若cfg中有cudnn_benchmark则Ture，默认False
    if cfg.get('cudnn_benchmark', False):  # get() 函数返回指定键的值，如果键不在字典中返回默认值
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename,CLI：命令行界面
    # 从终端传入的work_dir
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    # 当cfg中 work_dir 为 None 时，使用 ./work_dir/配置文件名/时间 作为默认工作目录
    elif cfg.get('work_dir', None) is None:  # get() 函数返回指定键的值，如果键不在字典中返回默认值
        # use config filename as default work_dir if cfg.work_dir is None
        # os.path.splitext(path) 分割路径，返回路径名和文件扩展名的元组
        # os.path.basename(path)返回path最后的文件名
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0],
                                time.strftime('%Y-%m-%d %X', time.localtime()))  # work_dir文件夹绝对路径
    # 断点续训
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)  # (0,1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]  # default=0

    # 根据GPU数量自动缩放lr
    if args.autoscale_lr:  # Ture
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        # 单卡lr = 总lr * gpu数量 / 8
        # 不改变单卡batch_size时，总lr不需要根据gpu数量调整，下面的公式已经自动调整单卡lr
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    # launcher 为 none，不启用分布式训练。不使用 dist_train.sh，default launcher=none
    if args.launcher == 'none':
        distributed = False
    # launcher 不为 none，启用分布式训练。使用 dist_train.sh，会传 ‘pytorch’
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))  # osp.abspath返回绝对路径，创建新的work_dir文件夹

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    # 获取时间并编码，作为文件名
    timestamp = time.strftime('%Y-%m-%d %X', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')  # 创建日志文件
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log env info
    env_info_dict = collect_env()  # 获取环境
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])  # 环境字典
    dash_line = '-' * 60 + '\n'  # 分隔线+换行 ------------------------------
    # logger.info环境输出流
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    meta['env_info'] = env_info  # 环境字典作为value写入meta
    meta['config'] = cfg.pretty_text  # 配置文件作为value写入meta

    # log some basic info 输出重要配置
    logger.info(f'Distributed training: {distributed}')  # 分布式训练
    logger.info(f'Config:\n{cfg.pretty_text}')  # 配置文件内容

    # set random seeds
    seed = init_random_seed(args.seed)  # 初始化随机种子
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # --------------------------------2.创建模型-----------------------------
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()  # 参数权重初始化
    # 输出模型信息（dict）
    logger.info(f'Model:\n{model}')

    # --------------------3.构建数据集: 需要传入 cfg.data.train，表明是训练集----------------------
    datasets = [build_dataset(cfg.data.train)]  # build训练配置和pipeline

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)  # 深拷贝一份训练时的验证配置
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES


    # -----------------------------4.将模型、数据、配置传入检测器训练--------------------------------
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
