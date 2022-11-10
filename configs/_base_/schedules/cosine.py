# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 0.003  # max learning rate
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,  # 当batch size增大时，应适当减小预热迭代次数
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)  # 最小学习率0.00001

momentum_config = None

runner = dict(type='EpochBasedRunner', max_epochs=40)  # 训练epoch
