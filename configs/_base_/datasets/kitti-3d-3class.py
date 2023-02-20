# dataset settings
dataset_type = 'KittiDataset'
data_root = '/home/rui/dataset/kitti/'  # 数据集根目录
class_names = ['Pedestrian', 'Cyclist', 'Car']
# 遵循右手系坐标：大拇指指向x轴的正方向，食指指向y轴的正方向时，中指微屈所指的方向就是z轴的正方向
# 激光雷达坐标系：x正方向为前，y正方向为左，z正方向为上
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # 裁减原始点云，选择合适的区域作为输入 x:0~70.4 y:-40~40 z：-3～1
input_modality = dict(use_lidar=True, use_camera=False)  # 输入数据模态，默认只使用雷达数据

file_client_args = dict(backend='disk')  # 文件从硬盘读入（默认所有信息都从硬盘输入，若从其他后端输入则需要启用下面注释的代码）
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/kitti/':
#         's3://openmmlab/datasets/detection3d/kitti/',
#         'data/kitti/':
#         's3://openmmlab/datasets/detection3d/kitti/'
#     }))

"""
    目标(car,pedestrian,cyclist)点云采样输入，目的是过滤掉极少点数的目标
    db_sampler (dict): Config dict of the database sampler.
"""
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',  # 数据预处理生成的训练文件
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],  # 选择是否过滤简单/中等/困难的目标
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),  # 少于多少个点的目标直接过滤掉，不输入参与训练
    classes=class_names,
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),  # 每种目标分为多少组采样 (在pointnet中会分组采样)
    # 点云数据加载
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',  # 坐标类型
        load_dim=4,  # x,y,z,intensity  加载维度
        use_dim=4,  # x,y,z,intensity  加载维度
        file_client_args=file_client_args),  # 点云数据从硬盘加载
    file_client_args=file_client_args)

"""
    训练过程中数据流，最终只剩下points,gt_bboxes_3d,gt_labels_3d
"""
train_pipeline = [
    # 加载原始points
    # 添加：points
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',  # 坐标类型 LIDAR\DEPTH\CAMERA
        load_dim=4,  # x,y,z,intensity  加载维度
        use_dim=4,  # x,y,z,intensity  使用维度
        file_client_args=file_client_args),

    # 添加：gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, pts_instance_mask, pts_semantic_mask, bbox3d_fields, pts_mask_fields, pts_seg_fields
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,  # 加载3d bbox
        with_label_3d=True,  # 加载3d标签
        file_client_args=file_client_args),

    # Sample GT objects to the data
    dict(type='ObjectSample', db_sampler=db_sampler),  # 24-40行定义目标点云采样

    # Apply noise to each GT objects in the scene
    dict(
        type='ObjectNoise',
        num_try=100,  # numbers of try 若噪音添加失败，则重试加入噪音
        translation_std=[1.0, 1.0, 0.5],  # 噪音标准偏差分布
        global_rot_range=[0.0, 0.0],  # 全局旋转
        rot_range=[-0.78539816, 0.78539816]),  # 目标旋转角度

    # Flip the points & bbox.
    # 添加：flip, pcd_horizontal_flip, pcd_vertical_flip 默认不翻转
    # 更新：points, *bbox3d_fields
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),  # 水平方向翻转points&bbox，ratio=0.5

    # Apply global rotation, scaling and translation to a 3D scene
    # 添加：pcd_trans, pcd_rotation, pcd_scale_factor
    # 更新：points, *bbox3d_fields
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],  # Range of rotation angle. (close to [-pi/4, pi/4]).
        scale_ratio_range=[0.95, 1.05]),  # Range of scale ratio.

    # Filter points by the range.去除区域外的点
    # 更新：points
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),

    # Filter objects by the range 去除区域外的3d框和标签
    # 更新：gt_bboxes_3d, gt_labels_3d
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),

    # Shuffle input points 去除区域外的点后，给点随机排序
    # 更新：points
    dict(type='PointShuffle'),

    # DefaultFormatBundle3D & Collect3D 格式化
    # 更新：points, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels
    dict(type='DefaultFormatBundle3D', class_names=class_names),  # mmdet3d/datasets/pipelines/formating.py-174行

    # 从特定任务加载流程中获取数据
    # 最后一个流程，决定哪些键值对应的数据会被输入给检测器
    # 添加：img_meta （由 meta_keys 指定的键值构成的 img_meta）
    # 移除：所有除 keys 指定的键值以外的其他键值
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

"""
    测试过程数据流，最后只输出points，并且作数据增强，目的是扩充验证集
"""
test_pipeline = [
    # 加载原始点云
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),

    # multiple scales and flipping
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,  # 点的缩放
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            # DefaultFormatBundle3D & Collect3D 格式化
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
"""
    验证过程数据流，最后只输出points，不做数据增强，这里的验证是指可视化，并不是验证
"""
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),

    # DefaultFormatBundle3D & Collect3D 格式化
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

"""
    数据集设置
"""
data = dict(
    samples_per_gpu=6,  # batch size  注意这里是每个GPU，当多个GPU时，不需要调整BS
    workers_per_gpu=4,  # 数据读取进程数

    train=dict(
        type='RepeatDataset',
        times=2,  # repeat times
        dataset=dict(
            type=dataset_type,  #  str 'KittiDataset'
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',  # 训练数据信息文件
            split='training',  # 数据集用途
            pts_prefix='velodyne_reduced',  # 训练数据位置，kitii经预处理程序后得到的点云文件夹
            pipeline=train_pipeline,  # 训练流程，在上面train_pipeline字典中定义
            modality=input_modality,  # default {use_lidar=True, use_camera=False}
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            file_client_args=file_client_args)),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        #samples_per_gpu=2  # 多batch_size验证
        ann_file=data_root + 'kitti_infos_val.pkl',  # 验证数据信息文件
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,  # 相较于验证流程，测试流程中做了数据增强
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

evaluation = dict(interval=1, pipeline=eval_pipeline)  # 训练过程中验证 每隔interval次epoch验证一次
