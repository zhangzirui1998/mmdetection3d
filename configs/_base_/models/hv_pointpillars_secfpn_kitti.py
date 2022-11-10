# point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]  单位m
voxel_size = [0.16, 0.16, 4]  # [x, y, z] -> [69.12, 79.36, 4]/[x, y, z] = [432, 496, 1]
"""
    class VoxelNet(SingleStage3DDetector):
        input: voxel_layer, voxel_encoder, middle_encoder, backbone, neck, bbox_head, 
                train_cfg, test_cfg, init_cfg, pretrained
"""
model = dict(
    type='VoxelNet',
    # 点云体素化
    voxel_layer=dict(
        max_num_points=32,  # max_points_per_voxel second=5,voxelnet=35
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)  # (training, testing) max_voxels  432*496=214272 有许多空体素，用稀疏卷积
    ),
    # 体素特征提取
    # input: [16000,32,10] pillars,points,features
    # output: [16000,64]
    voxel_encoder=dict(
        type='PillarFeatureNet',  # prepares the pillar features and performs forward pass through PFNLayers(特征金字塔)
        in_channels=4,  # Number of input features: x,y,z,r
        feat_channels=[64], # Number of features in each of the N PFNLayers
        with_distance=False,  # Whether to include Euclidean distance to points
        # with_cluster_center=True, with_voxel_center=True (default)
        voxel_size=voxel_size,  # mode=max (pooling), norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01) (default)
        # second源码中用avg pooling
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    # Converts learned features from dense tensor to sparse pseudo image
    # input: in_channels=64, output_shape=[496, 432]
    # output: 伪图像[1,64,432,496]：[pillar,features,y,x] ? 为什么输出将 x,y 互换位置
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    # 输入伪图像，提取多尺度特征，然后输出多尺度伪图像特征  多尺度=3
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],  #  Number of layers in each stage 每一个stage含有的卷积层数量（不同stage是不同尺度的特征）
        layer_strides=[2, 2, 2],  # Strides of each stage
        out_channels=[64, 128, 256]),  # Output channels for multi-scale feature maps
    # 通过特征金字塔将多尺度特征融合
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],  # Input channels of multi-scale feature maps
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),  # Output channels of feature maps
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,  # 128*3 Number of channels in the input feature map
        feat_channels=384,  # Number of channels of the feature map
        use_direction_classifier=True,  #  Whether to add a direction classifier.
        assign_per_class=True,  #  Whether to do assignment for each class
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',  # Aligned 3D Anchor Generator by range
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],  # x_min, y_min, z_min, x_max, y_max, z_max
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],  # Anchor size x,y,z
            rotations=[0, 1.57],  # Rotations of anchors in a single feature grid.
            reshape_out=False),
        diff_rad_by_sin=True,  # Whether to change the difference into sin difference for box regression loss.
        # Bbox Coder for 3D boxes
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),  # 对偏差和真实值的计算进行编码
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,  # 用于抑制容易分辨的样本的损失
            alpha=0.25,  # 用于处理正负样本不平衡，即正样本要比负样本占比小，这是因为负例易分
            loss_weight=1.0),  # 分类损失占总损失的权重
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),  # 先验分布服从拉普拉斯
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),  # 注意此处不做sigmoid,因为是回归
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),  # Nearest 3D IoU Calculator
                pos_iou_thr=0.5,  # IoU threshold for positive bboxes
                neg_iou_thr=0.35,  # IoU threshold for negative bboxes
                min_pos_iou=0.35,  # Minimum iou for a bbox to be considered as a positive bbox
                # gt_max_assign_all=True，意思是当一个gt与他的所有anchor的iou都不超过0.5，
                # 那么就把与gt的iou最高的所有anchor都设为postive,但这个iuo的阈值为min_pos_iou
                ignore_iof_thr=-1),  # ignore_iof_thr为负值意思是不忽略任何bbox
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,  # 计算旋转时使用非极大值抑制
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
