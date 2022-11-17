# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class SECOND(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,  # pp in_channels=64
                 out_channels=[128, 128, 256],  #pp out_channels=[64, 128, 256]
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),  # 若在卷积或者神经网络计算后加了正则化项，则不需要bias，节省内存
                 init_cfg=None,  # 可能是权重初始化的意思
                 pretrained=None):
        super(SECOND, self).__init__(init_cfg=init_cfg)
        # 保证输入、输出、卷积层有相同的stage
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        # 不明白这里为什么要取out_channels的前两位——根据原文图猜测是将上一个stage的输出作为下一个stage的输入，在下面创建卷积层的代码中可以看出
        in_filters = [in_channels, *out_channels[:-1]]  # *out_channels[:-1]列表取值到倒数第二位，再依次输出值 in_filters=[64,64,128]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []  # 卷积池化等操作放在一个块里
        for i, layer_num in enumerate(layer_nums):  # enumerate 返回索引和值
            block = [
                # 创建卷积层：先将每个stage输入的特征维度升维成输出维度64-64，64-128，128-256
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],  # 64,64,128
                    out_channels[i],  # 64,128,256
                    3,  # 3*3conv
                    stride=layer_strides[i],  # 2,2,2
                    padding=1),  # 3*3conv,p=1，s=2 输出特征图大小比输入小一半
                build_norm_layer(norm_cfg, out_channels[i])[1],  # 每次卷积操作后都作一次BN
                nn.ReLU(inplace=True),  # 每次卷积操作后都作一次ReLU
            ]
            for j in range(layer_num):  # 3,5,5 对每个stage作不同次数的卷积
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        # （64-64）*3 （128-128）*5  （256-256）*5
                        out_channels[i],  # 64,128,256
                        out_channels[i],  # 64,128,256
                        3,
                        padding=1))  # 3*3conv,p=1，s=1 输入输出特征图大小一样
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])  # 每次卷积操作后都作一次BN
                block.append(nn.ReLU(inplace=True))  # 每次卷积操作后都作一次ReLU

            block = nn.Sequential(*block)  # 将列表block的值依次传给nn.Sequential，数据格式转换
            blocks.append(block)  # 每个stage的模块放在一个block里面

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'  # 权重初始化和预训练权重不能同时加载
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '  # 预训练不推荐
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')  # 用凯明的初始化配置

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).  N:points number C:channels H,W:voxel size

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):  # len(self.blocks)=3
            # self.blocks是Modulelist要写forward，block是Sequential自带forward
            x = self.blocks[i](x)  # 将参数x传入nn.module中进行计算
            outs.append(x)  # 输出是伪图像的特征
        return tuple(outs)
