# model settings
voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=35,  # max_points_per_voxel second=5,voxelnet=35
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),  # (training, testing) max_voxels  432*496=214272 有许多空体素，用稀疏卷积
    voxel_encoder=dict(
        type='AttnPFN',
        in_channels=4,
        feat_channels=(64,),
        with_distance=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        mode='max',  # second源码中用avg pooling
        multihead_attention=dict(
            type='SelfAttention',
            num_attention_heads=4,
            input_size=11,
            hidden_size=64)),  # out_channels=64
    # output: 伪图像[1,64,496,432]：[pillar,features,y,x] ? 为什么输出将 x,y 互换位置
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='RCSECOND',
        in_channels=[64, 128, 256],
        out_channels=[128, 256, 512],
        fpn_channels=[64, 128, 256],
        layer_nums=[8, 5, 2],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        init_cfg=None,  # 权重初始化
        pretrained=None
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=384,  # 128*3 Number of channels in the input feature map
        feat_channels=384,  # Number of channels of the feature map
        use_direction_classifier=True,  #  Whether to add a direction classifier.
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',  # Aligned 3D Anchor Generator by range
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],  # x_min, y_min, z_min, x_max, y_max, z_max
            sizes=[[3.9, 1.6, 1.56]],  # Anchor size x,y,z
            rotations=[0, 1.57],  # Rotations of anchors in a single feature grid.
            reshape_out=True),
        diff_rad_by_sin=True,  # 是否使用角度差的正弦值作为损失
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),  # 对偏差和真实值的计算进行编码 anchor based
        loss_cls=dict(
            type='FocalLoss',  # 由于使用了FocalLoss，不需要对正负样本作平衡
            use_sigmoid=True,
            gamma=2.0,  # 用于抑制容易分辨的样本的损失
            alpha=0.25,  # 用于处理正负样本不平衡，即正样本要比负样本占比小，这是因为负例易分
            loss_weight=1.0),  # 分类损失占总损失的权重
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),  # 先验分布服从拉普拉斯
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),  # 注意此处不做sigmoid,因为是回归
    # model training and testing settings
    train_cfg=dict(
        # 正负样本分类
        assigner=dict(
            type='MaxIoUAssigner',  # anchor based
            iou_calculator=dict(type='BboxOverlapsNearest3D'),  # Nearest 3D IoU Calculator
            pos_iou_thr=0.6,  # IoU threshold for positive bboxes
            neg_iou_thr=0.45,  # IoU threshold for negative bboxes
            min_pos_iou=0.45,  # Minimum iou for a bbox to be considered as a positive bbox
            # gt_max_assign_all=True，意思是当一个gt与他的所有anchor的iou都不超过0.6，
            # 那么就把与gt的iou最高的所有anchor都设为postive,但这个iuo的阈值为min_pos_iou
            ignore_iof_thr=-1),  # ignore_iof_thr为负值意思是不忽略任何bbox
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,  # 计算旋转时使用非极大值抑制
        nms_across_levels=False,
        nms_thr=0.01,  # 3D框的IOU超过0.01则做NMS
        score_thr=0.1,  # 3D框的置信度小于0.1则做NMS
        min_bbox_size=0,
        nms_pre=100,  # 取出置信度前100的预测框做NMS
        max_num=50))  # 做完NMS后只保留最终的50个框

# dataset settings
dataset_type = 'KittiDataset'
data_root = '/root/autodl-tmp/kitti/'
class_names = ['Car']
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    sample_groups=dict(Car=15),
    classes=class_names,
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    file_client_args=file_client_args)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    # dict(type='NormalizeIntensityTanh', intensity_column=3),  # 增加fp16中数值稳定性
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
            # dict(type='NormalizeIntensityTanh', intensity_column=3),  # 增加fp16中数值稳定性
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            file_client_args=file_client_args)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        samples_per_gpu=8,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args))
# optimizer
lr = 0.001
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99), # the momentum is change during training
    weight_decay=0.01)
# max_norm越大，对于梯度爆炸的解决越柔和，max_norm越小，对梯度爆炸的解决越狠
# norm_type越小，对梯度裁剪越厉害
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))   # 将梯度的模裁剪到35以内
# learning policy
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
checkpoint_config = dict(interval=4)
evaluation = dict(interval=4, pipeline=eval_pipeline)
# yapf:disable
log_config = dict(
    interval=70,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='Your-project'))
    ])
# yapf:enable
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=80)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
# fp16 = dict(loss_scale=32.)
# 检查loss异常值
checkinvalidloss = dict(type='CheckInvalidLossHook', interval=70)  # 每隔多少个iter检查一次