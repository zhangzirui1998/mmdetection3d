voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
model = dict(
    type='DynamicVoxelNet',
    voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(-1, -1)),
    voxel_encoder=dict(
        type='DynamicAttnPFN',
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max',  # second源码中用avg pooling
        legacy=True,
        multihead_attention=dict(
            type='DynamicSelfAttention',
            num_attention_heads=4,
            input_size=10,
            hidden_size=64),
        # mlp
        mlp_in_channel=64,
        mlp_conv_channels=(64,)
    ),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='RCSECOND',
        in_channels=[64, 128, 256],
        out_channels=[128, 256, 512],
        fpn_channels=[64, 128, 256],
        layer_nums=[5, 5, 1],
        layer_strides=[2, 2, 2]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True),  # 当只有一个类别时reshape=True
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
dataset_type = 'KittiDataset'
data_root ='/root/autodl-tmp/kitti/'
class_names = ['Car']
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root+'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15),
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
    # Sample GT objects to the data, use_ground_plane:Whether to use ground plane to adjust the 3D labels
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    # Apply noise to each GT objects in the scene
    dict(
        type='ObjectNoise',
        num_try=100,  # Number of times to try if the noise applied is invalid
        translation_std=[1, 1, 0.5],
        global_rot_range=[0.0, 0.0],  # Global rotation to the scene
        rot_range=[-0.15707963267, 0.15707963267]),  # Object rotation range
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],  # Range of rotation angle
        scale_ratio_range=[0.95, 1.05],  # Range of scale ratio
        translation_std=[0.25, 0.25, 0.25]),  # 应用于场景的平移噪声的标准偏差，从高斯分布采样，其标准偏差由“translation_std”设置
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),  # Filter points by the range
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),  # Filter objects by the range
    dict(type='PointShuffle'),  # Shuffle input points
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
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root+'kitti_infos_train.pkl',
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
        samples_per_gpu=2,
        ann_file=data_root+'kitti_infos_val.pkl',
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
        ann_file=data_root+'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args))

lr = 0.001
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
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
runner = dict(type='EpochBasedRunner', max_epochs=80)
evaluation = dict(interval=1, pipeline=eval_pipeline)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='Your-project'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
# fp16 = dict(loss_scale=32.)
# 检查loss异常值
checkinvalidloss = dict(type='CheckInvalidLossHook', interval=30)  # 每隔多少个iter检查一次
