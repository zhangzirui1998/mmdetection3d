# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from mmcv.runner import force_fp32, auto_fp16
from torch import nn

from ..builder import VOXEL_ENCODERS
from .utils import PFNLayer, get_paddings_indicator
from torch.nn.modules import activation
from torch.nn.modules.activation import MultiheadAttention
from ..model_utils.transformer import PillarAttention
import torch.nn.functional as F

@VOXEL_ENCODERS.register_module()
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0  # feat_channels=[64] from pp
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3  # 7
        if with_voxel_center:
            in_channels += 3  # 10
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance  # False
        self._with_cluster_center = with_cluster_center  # True
        self._with_voxel_center = with_voxel_center  # True
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels  # 10
        feat_channels = [in_channels] + list(feat_channels)  # [10,64]
        pfn_layers = []
        for i in range(len(feat_channels) - 1):  # i=0,pfn中只有一层网络
            in_filters = feat_channels[i]  # 10
            out_filters = feat_channels[i + 1]  # 64
            if i < len(feat_channels) - 2:  # False
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,  # 10
                    out_filters,  # 64
                    norm_cfg=norm_cfg,  # {'type': 'BN1d', 'eps': 0.001, 'momentum': 0.01}
                    last_layer=last_layer,  # True
                    mode=mode))  # max
        self.pfn_layers = nn.ModuleList(pfn_layers)  # fpn层加入模块中

        # Need pillar (voxel) size and x/y offset in order to calculate offset  计算体素中点的偏移量
        self.vx = voxel_size[0]  # 0.16
        self.vy = voxel_size[1]  # 0.16
        self.vz = voxel_size[2]  # 4
        self.x_offset = self.vx / 2 + point_cloud_range[0]  # 0.08
        self.y_offset = self.vy / 2 + point_cloud_range[1]  # -39.6
        self.z_offset = self.vz / 2 + point_cloud_range[2]  # -1.0
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C). = voxels
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.体素自身坐标，16000x4，[batch_id, x, y, z]

        Returns:
            torch.Tensor: Features of pillars.
        """
        # 将传入的特征tensor放入list
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            # points_mean体素内中心点坐标
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            # 体素中每个点与中心点坐标偏移
            f_cluster = features[:, :, :3] - points_mean
            # 加入偏移聚类中心特征
            features_ls.append(f_cluster)  # [[x,y,z,r],[x',y',z']]

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype  # torch.float32
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            # 加入偏移体素中心特征
            features_ls.append(f_center)  # [[x,y,z,r],[x',y',z'],[xp,yp,zp]]

        if self._with_distance:  # default False
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)  # concat特征 10 [x,y,z,r,x',y',z',xp,yp,zp]
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        # 将采样点不足的voxel中加入0补充
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # 抽取每个体素内的特征
        for pfn in self.pfn_layers:
            # 将特征放入FPN中提取并maxpooling，channels: in10 out64
            features = pfn(features, num_points)  # 32*10->32*64->1*64

        return features.squeeze(1)  # 去除tensor中维度为1的维度，此处就是去除num_points这个维度


@VOXEL_ENCODERS.register_module()
class DynamicPillarFeatureNet(PillarFeatureNet):
    """Pillar Feature Net using dynamic voxelization.

    The network prepares the pillar features and performs forward pass
    through PFNLayers. The main difference is that it is used for
    dynamic voxels, which contains different number of points inside a voxel
    without limits.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64,),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(DynamicPillarFeatureNet, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy)
        self.fp16_enabled = False
        feat_channels = [self.in_channels] + list(feat_channels)  # [10, 64]

        # 重新写pfn_layers
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(feat_channels) - 1):  # 只添加一层PFN
            in_filters = feat_channels[i]  # [10]
            out_filters = feat_channels[i + 1]  # [64]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,  # 10， 64
                    nn.LeakyReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)  # 1
        self.pfn_layers = nn.ModuleList(pfn_layers)
        # DynamicScatter(voxel_size=[0.16, 0.16, 4], point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1], average_points=False)
        self.pfn_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        # DynamicScatter(voxel_size=[0.16, 0.16, 4], point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1], average_points=True)
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map the centers of voxels to its corresponding points.
            将体素的中心映射到其对应点
        Args:
            pts_coors (torch.Tensor): The coordinates of each points, shape
                (M, 3), where M is the number of points.
            voxel_mean (torch.Tensor): The mean or aggregated features of a
                voxel, shape (N, C), where N is the number of voxels.
            voxel_coors (torch.Tensor): The coordinates of each voxel.

        Returns:
            torch.Tensor: Corresponding voxel centers of each points, shape
                (M, C), where M is the number of points.
        """
        # Step 1: scatter voxel into canvas 将体素分散到画布中
        # Calculate necessary things for canvas creation 计算伪图像大小
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)  # 496
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)  # 432
        canvas_channel = voxel_mean.size(1)  # 4 伪图像输入通道数
        batch_size = pts_coors[-1, 0] + 1  # 2
        canvas_len = canvas_y * canvas_x * batch_size  # 428544
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)  # (4,428544)
        # Only include non-empty pillars 体素索引
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])  # (98,)
        # Scatter the blob back to the canvas将体素特征放到伪图像上
        canvas[:, indices.long()] = voxel_mean.t()  # (4,428544)

        # Step 2: get voxel mean for each point
        # 点索引
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])  # (4030,)
        # 将体素特征传给体素内的点
        center_per_point = canvas[:, voxel_index.long()].t()  # (4030,4)
        return center_per_point  # 每个点对应的体素特征

    @force_fp32(out_fp16=True)
    def forward(self, features, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C). (4030,4)原始点云坐标和r
            coors (torch.Tensor): Coordinates of each voxel (4030,4)

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]  # 加入原始点云特征(4030,4) 4:(x,y,z,r)
        # Find distance of x, y, and z from cluster center 计算点和聚类中心的距离
        if self._with_cluster_center:
            # 计算聚类中心
            # coors:所有体素的坐标/所有点对应的体素坐标    mean_coors:含有points的体素坐标    voxel_mean：由points提取出的体素特征
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)  # (98,4) (98,4)
            # points_mean:计算点的聚类中心
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)  # (4030,4)
            # TODO: maybe also do cluster for reflectivity
            # 计算点和点聚类中心的偏差
            f_cluster = features[:, :3] - points_mean[:, :3]  # (4030,3)
            features_ls.append(f_cluster)  # [(4030,4),(4030,3)]

        # Find distance of x, y, and z from pillar center 计算点和体素中心的距离
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))  # (4030,3)
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)  # 点x方向的偏移
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)  # 点y方向的偏移
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)  # 点z方向的偏移
            features_ls.append(f_center)  # f_center=(4030,3) features_ls=[(4030,4),(4030,3)]

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)  # (4030,10) include:点特征(原始坐标)，点与聚类中心距离，点和体素中心距离
        for i, pfn in enumerate(self.pfn_layers):
            # 在全局点云内提取点的特征，得到每个点的全局特征 10->64
            point_feats = pfn(features)  # (4030,64)
            # 由体素内的点特征做最大池化得到体素特征 voxel_feats:体素特征 voxel_coors:体素坐标
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)  # (98,64) (98,4)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        return voxel_feats, voxel_coors  # 提取后的体素特征和体素坐标



# 定义新的PFN层，不做最大值操作，保留每个Voxel的全部特征点
class PFNLayer_nomax(nn.Module):
    def __init__(self,
                 in_channels,  # 10
                 out_channels,  # 64
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),  # {'type':'BN1d', 'eps':0.001, 'momentum':0.01}
                 last_layer=False,  # True
                 mode='max'):  # max
        super(PFNLayer_nomax, self).__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer_nomax'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels  # 64
        # BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        # Linear(in_features=10, out_features=64, bias=False)
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg']
        self.mode = mode

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        '''

        Args:
            inputs: (N, M, C)
            num_voxels:

        Returns:
            torch.Tensor: Features of Pillars.
        '''
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.leaky_relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                -1, 1, 1)

        if self.last_vfe:
            return x
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@VOXEL_ENCODERS.register_module()
class AttnPFN(nn.Module):
    def __init__(self,
                 in_channels=4,
                 feat_channels=(64,),
                 embed_dims=3,
                 num_heads=3,
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(AttnPFN, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        assert len(feat_channels) > 0  # feat_channels=[64] from pp
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3  # 7
        if with_voxel_center:
            in_channels += 3  # 10
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance  # False
        self._with_cluster_center = with_cluster_center  # True
        self._with_voxel_center = with_voxel_center  # True
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels  # 10
        feat_channels = [in_channels] + list(feat_channels)  # [10,64]
        pfn_layers = []
        for i in range(len(feat_channels) - 1):  # i=0,pfn中只有一层网络
            in_filters = feat_channels[i]  # 10
            out_filters = feat_channels[i + 1]  # 64
            if i < len(feat_channels) - 2:  # False
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer_nomax(
                    in_filters,  # 10
                    out_filters,  # 64
                    norm_cfg=norm_cfg,  # {'type': 'BN1d', 'eps': 0.001, 'momentum': 0.01}
                    last_layer=last_layer,  # True
                    mode=mode))  # max
        self.pfn_layers = nn.ModuleList(pfn_layers)  # fpn层加入模块中

        # Need pillar (voxel) size and x/y offset in order to calculate offset  计算体素中点的偏移量
        self.vx = voxel_size[0]  # 0.16
        self.vy = voxel_size[1]  # 0.16
        self.vz = voxel_size[2]  # 4
        self.x_offset = self.vx / 2 + point_cloud_range[0]  # 0.08
        self.y_offset = self.vy / 2 + point_cloud_range[1]  # -39.6
        self.z_offset = self.vz / 2 + point_cloud_range[2]  # -1.0
        self.point_cloud_range = point_cloud_range


    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

                Args:
                    features (torch.Tensor): Point features or raw points in shape
                        (N, M, C). = voxels
                    num_points (torch.Tensor): Number of points in each pillar.
                    coors (torch.Tensor): Coordinates of each voxel.体素自身坐标，16000x4，[batch_id, x, y, z]

                Returns:
                    torch.Tensor: Features of pillars.
                """
        # 通过注意力模块计算每个点的最大注意力分数
        attn_block = PillarAttention(self.embed_dims, self.num_heads)
        features_attn = attn_block(features[:,:,:3])
        # TODO：将相关性分数做归一化，控制值在0-1之间
        max_features_attn = torch.max(features_attn, -1, keepdim=True)[0]  # 作为每个点的相关性分数

        # 用PFN层提取特征
        # 将传入的特征tensor放入list
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            # points_mean体素内中心点坐标
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                -1, 1, 1)
            # 体素中每个点与中心点坐标偏移
            f_cluster = features[:, :, :3] - points_mean
            # 加入偏移聚类中心特征
            features_ls.append(f_cluster)  # [[x,y,z,r],[x',y',z']]

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype  # torch.float32
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                        coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                        self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                        coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                        self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                        coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                        self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                        coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                        self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                        coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                        self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                        coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                        self.z_offset)
            # 加入偏移体素中心特征
            features_ls.append(f_center)  # [[x,y,z,r],[x',y',z'],[xp,yp,zp]]

        if self._with_distance:  # default False
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)  # concat特征 10 [x,y,z,r,x',y',z',xp,yp,zp]
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        # 将采样点不足的voxel中加入0补充
        voxel_count = features.shape[1]  # 每个体素中的点数
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # 抽取每个体素内的特征
        for pfn in self.pfn_layers:
            # 将特征放入FPN中提取并maxpooling，channels: in10 out64
            features = pfn(features, num_points)  # 32*10->32*64->1*64

        # 每个点的最大注意力分数与PFN提取的点特征相乘再除总注意力分数（正则化）
        fa = (max_features_attn * features) / max_features_attn.sum(dim=1, keepdim=True)
        fa = torch.max(fa, dim=1, keepdim=True)[0]
        # 经过FPN后再做max的特征
        fm = torch.max(features, dim=1, keepdim=True)[0]
        # 总特征
        f = (fa + fm) / 2

        return f.squeeze(1)
