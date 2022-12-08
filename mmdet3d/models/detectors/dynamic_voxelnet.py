# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from ..builder import DETECTORS
from .voxelnet import VoxelNet


@DETECTORS.register_module()
class DynamicVoxelNet(VoxelNet):
    r"""VoxelNet using `dynamic voxelization
        <https://arxiv.org/abs/1910.06528>`_.
    """

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        '''
        eg. xmu_fusion/dv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py
        Args:
            voxel_layer:{'max_num_points': -1, 'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1], 'voxel_size': [0.16, 0.16, 4], 'max_voxels': (-1, -1)}
            voxel_encoder:{'type': 'DynamicPillarFeatureNet', 'in_channels': 4, 'feat_channels': [64], 'with_distance': False, 'voxel_size': [0.16, 0.16, 4], 'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1]}
            middle_encoder:{'type': 'PointPillarsScatter', 'in_channels': 64, 'output_shape': [496, 432]}
            backbone:{'type': 'SECOND', 'in_channels': 64, 'layer_nums': [3, 5, 5], 'layer_strides': [2, 2, 2], 'out_channels': [64, 128, 256]}
            neck:{'type': 'SECONDFPN', 'in_channels': [64, 128, 256], 'upsample_strides': [1, 2, 4], 'out_channels': [128, 128, 128]}
        '''
        super(DynamicVoxelNet, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def extract_feat(self, points, img_metas):  # img_metas=None
        """Extract features from points."""
        # 没有规定每个voxel的点数量
        # voxels:orign point features(coors) in shape(N,4). N is the number of points,4:(x,y,z,r)
        # coors:Coordinates of voxels shape(N,4). N is the number of points,4:(batch_id,z,y,x)
        voxels, coors = self.voxelize(points)  # 体素化 (4030,4),(4030,4)
        # voxel_features:Mean of points inside each voxel in shape (M, 3(4))  M is the number of voxels(不是原voxel数量)
        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)  # (98,64),(98,4)
        batch_size = coors[-1, 0].item() + 1  # 2
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)  # (2,64,496,432)
        x = self.backbone(x)  # ((2,64,248,216),(2,128,124,108),(2,256,62,54))
        # 判断是否包含neck模块
        if self.with_neck:
            x = self.neck(x)  # [(2,384,248,216)]
        return x  # [(2,384,248,216)]

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
            动态体素化是将点和其所在的体素坐标做一个映射
        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []  # [(2010,3),(2020,3)]
        # dynamic voxelization only provide a coors mapping 动态体素化只是提供一个体素坐标映射
        # 对points(list[tensor])进行迭代，返回值为tensor. test:points=[(2010,4),(2020,4)]
        # self.voxel_layer()实现对点的体素化，并计算体素坐标
        for res in points:
            # res_coors= tensor:(2010, 3)\(2020, 3) res_coors是每个点对应的动态voxel的坐标
            res_coors = self.voxel_layer(res)  # [M,(z,y,x)]
            coors.append(res_coors)
        points = torch.cat(points, dim=0)  # (4030,4) 4:(x,y,z,r)
        # 给coors增加batch_id
        coors_batch = []
        for i, coor in enumerate(coors):
            # 在第一个位置增加一个维度，并令值为i，将coors从3个维度增加到4个维度.增加的维度为batch_id
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)  # (4030,4) 4:(batch_id,z,y,x)
        return points, coors_batch  # (4030,4),(4030,4)
