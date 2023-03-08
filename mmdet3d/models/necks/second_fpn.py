# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from ..builder import NECKS


@NECKS.register_module()
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],  # pp in_channels=[64, 128, 256]
                 out_channels=[256, 256, 256],  # pp out_channels=[128, 128, 128]
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []  # 反卷积块
        for i, out_channel in enumerate(out_channels):  # 128,128,128
            stride = upsample_strides[i]  # 1,2,4
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                # 构建上采样层
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],  # 64，128，256
                    out_channels=out_channel,  # 128，128，128
                    kernel_size=upsample_strides[i],  # 1，2，4
                    stride=upsample_strides[i])  # 1，2，4
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            # deblock:Sequential
            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.LeakyReLU(inplace=True))
            # deblocks:list
            deblocks.append(deblock)
        # self.deblocks:ModuleList
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.[(2,64,248,216),(2,128,124,108),(2,256,62,54)]

        Returns:
            list[torch.Tensor]: Multi-level feature maps.多个同一大小特征图
        """
        assert len(x) == len(self.in_channels)
        # 反卷积上采样.将backbone输出的不同大小特征图上采样一样大
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]  # [(2,128,248,216),(2,128,248,216),(2,128,248,216)]

        if len(ups) > 1:  # 3
            out = torch.cat(ups, dim=1)  # (2,384,248,216) 融合特征图
        else:
            out = ups[0]  # ups[0]=N
        return [out]  # [(2,384,248,216)]


@NECKS.register_module()
class RCSECONDFPN(BaseModule):
    '''
    上采样倍数1-2-4
    '''
    def __init__(self,
                 in_channels=[64, 128, 256],  # pp in_channels=[64, 128, 256]
                 out_channels=[128, 128, 128],  # pp out_channels=[128, 128, 128]
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None
                 ):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(RCSECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        # 反卷积模块
        deblocks = []  # 反卷积块
        for i, out_channel in enumerate(out_channels):  # 128,128,128
            stride = upsample_strides[i]  # 1,2,4
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                # 构建上采样层
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=out_channel,  # 128，128，128
                    out_channels=out_channel,  # 128，128，128
                    kernel_size=upsample_strides[i],  # 1，2，4
                    stride=upsample_strides[i])  # 1，2，4
            else:
                # 普通卷积层
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            # deblock:Sequential
            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.LeakyReLU(inplace=True))
            # deblocks:list
            deblocks.append(deblock)
        # self.deblocks:ModuleList
        self.deblocks = nn.ModuleList(deblocks)

        # 构建1*1升维卷积层 [64, 128, 256]->[128，128，128]
        blocks = []
        for j, in_channel in enumerate(in_channels):  # [64, 128, 256]
            block = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=in_channel, out_channels=out_channels[j],
                                 kernel_size=1, stride=1),
                build_norm_layer(norm_cfg, out_channels[j])[1],
                nn.LeakyReLU(inplace=True)
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]


    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.[(2,64,248,216),(2,128,124,108),(2,256,62,54)]

        Returns:
            list[torch.Tensor]: Multi-level feature maps.多个同一大小特征图
        """
        assert len(x) == len(self.in_channels)
        # 特征图大小不变，维度变为128，128，128
        x = [block(x[j]) for j, block in enumerate(self.blocks)]  # [(2,128,248,216),(2,128,124,108),(2,128,62,54)]

        # 反卷积上采样.将backbone输出的不同大小特征图上采样一样大*1,*2,*4
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]  # [(2,128,248,216),(2,128,248,216),(2,128,248,216)]

        if len(ups) > 1:  # 3
            out = torch.cat(ups, dim=1)  # (2,384,248,216) 融合特征图
        else:
            out = ups[0]  # ups[0]=N
        return [out]  # [(2,384,248,216)]

