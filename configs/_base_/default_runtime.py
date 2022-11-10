# 权重文件配置
checkpoint_config = dict(interval=1)  # 每隔interval次保存一次权重
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,  # 每隔interval次保存一次日志
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='Your-project'))
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None  # 工作空间
load_from = None
resume_from = None  # 断点续训
# 工作流程配置
workflow = [('train', 1)]  # 运行 1 个 epoch 进行训练
# workflow = [('train', 1), ('val', 1)]  1 个 epoch 的训练和 1 个 epoch 的验证将被迭代运行
# workflow = [('val', 1), ('train', n)]  1 个 epoch 的验证和 1 个 epoch 的训练将被迭代运行
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
