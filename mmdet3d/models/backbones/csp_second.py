import warnings
import torch
import numpy as np
from ..builder import BACKBONES
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn


@BACKBONES.register_module()
class RCSECOND(BaseModule):
    def __init__(self,
                 in_channels=[64, 128, 256],
                 out_channels=[128, 256, 512],
                 fpn_channels=[64, 128, 256],
                 layer_nums=[5, 5, 1],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,  # 权重初始化
                 pretrained=None):

        super(RCSECOND, self).__init__(init_cfg=init_cfg)
        # 保证输入、输出、卷积层有相同的stage
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        self.fp16_enabled = False
        self.layer_nums = layer_nums
        # LeakyReLU
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

        # CSP net
        csp_in_channels = out_channels  # [128,256,512]
        csp_out_channels = np.array(csp_in_channels)/2
        csp_out_channels = csp_out_channels.astype(dtype=int).tolist()  # [64,128,256]
        rep_channels = csp_out_channels  # [64,128,256]

        # 创建卷积进行维度切分
        # 1.进入rep_conv部分
        block1 = []
        for i in range(len(csp_in_channels)):
            block1.append(nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    csp_in_channels[i],  # [128,256,512]
                    csp_out_channels[i],  # [64,128,256]
                    1,
                    stride=1,
                    padding=0),
                build_norm_layer(norm_cfg, csp_out_channels[i])[1],  # [64,128,256]
                nn.LeakyReLU(inplace=True))
            )
        self.block1 = nn.ModuleList(block1)
        # 2.进入csp部分
        block2 = []
        for j in range(len(csp_in_channels)):
            block2.append(nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    csp_in_channels[j],  # [128,256,512]
                    csp_out_channels[j],  # [64,128,256]
                    1,
                    stride=1,
                    padding=0),
                build_norm_layer(norm_cfg, csp_out_channels[j])[1],
                nn.LeakyReLU(inplace=True))
            )
        self.block2 = nn.ModuleList(block2)

        # repconv
        self.downsample_block = nn.ModuleList()
        self.rep_block = nn.ModuleList()
        for n, layer_num in enumerate(layer_nums):
            rep_blocks = []
            downsample_block = [
                nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        in_channels[n],  # [64,128,256]
                        out_channels[n],  # [128,256,512]
                        3,
                        stride=layer_strides[n],
                        padding=1),  # 2倍下采样
                    build_norm_layer(norm_cfg, out_channels[n])[1]),
                nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        in_channels[n],  # [64,128,256]
                        out_channels[n],  # [128,256,512]
                        1,
                        stride=layer_strides[n],
                        padding=0),  # 2被下采样
                    build_norm_layer(norm_cfg, out_channels[n])[1])
            ]
            self.downsample_block.append(nn.ModuleList(downsample_block))
            for m in range(layer_num):  # 3,5,5 对每个stage作不同次数的卷积
                rep_block = [
                    nn.Sequential(
                        build_conv_layer(
                            conv_cfg,
                            rep_channels[n],  # [64,128,256]
                            rep_channels[n],  # [64,128,256]
                            3,
                            stride=1,
                            padding=1),  # [64,128,256]
                        build_norm_layer(norm_cfg, rep_channels[n])[1]),
                    nn.Sequential(
                        build_conv_layer(
                            conv_cfg,
                            rep_channels[n],  # [64,128,256]
                            rep_channels[n],  # [64,128,256]
                            1,
                            stride=1,
                            padding=0),  # [64,128,256]
                        build_norm_layer(norm_cfg, rep_channels[n])[1]),
                    nn.Sequential(build_norm_layer(norm_cfg, rep_channels[n])[1])]
                rep_blocks.append(nn.ModuleList(rep_block))
            self.rep_block.append(nn.ModuleList(rep_blocks))

        # 普通conv,bn,act层
        self.final = nn.ModuleList()
        for t in range(len(out_channels)):
            self.final.append(
                nn.Sequential(
                    build_conv_layer(conv_cfg, out_channels[t], out_channels[t], 1),
                    build_norm_layer(norm_cfg, out_channels[t])[1],
                    nn.LeakyReLU(inplace=True))
            )

        # 降维输入FPN
        self.down_dim = nn.ModuleList()
        for k in range(len(fpn_channels)):
            self.down_dim.append(
                nn.Sequential(
                    build_conv_layer(conv_cfg, out_channels[k], fpn_channels[k], 1),
                    build_norm_layer(norm_cfg, fpn_channels[k])[1],
                    nn.LeakyReLU(inplace=True)
                )
            )

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'  # 权重初始化和预训练权重不能同时加载
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '  # 预训练不推荐
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')  # 用凯明的初始化配置

    def forward(self, x):
        rep_outs = []
        outs = []

        for i, layer_nums in enumerate(self.layer_nums):  # 二维列表求长度，默认返回第一个维度
            # down sample
            x1 = self.downsample_block[i][0](x)  # 3*3
            x2 = self.downsample_block[i][1](x)  # 1*1
            x = self.LeakyReLU(x1+x2)
            # csp_net
            x_rep = self.block1[i](x)
            x_id = self.block2[i](x)
            # rep_conv
            y1, y2, y3 = x_rep, x_rep, x_rep
            for m in range(layer_nums):
                y1 = self.rep_block[i][m][0](y1)  # 3*3
                y2 = self.rep_block[i][m][1](y2)  # 1*1
                y3 = self.rep_block[i][m][2](y3)  # bn
                y = self.LeakyReLU(y1+y2+y3)
                y1, y2, y3 = y, y, y

            x = torch.cat((x_id, y), dim=1)
            x = self.final[i](x)  # 1*1
            rep_outs.append(x)  # [2,128,248,216],[2,256,124,108],[2,512,64,54]
        # 降维/2
        for j in range(len(self.down_dim)):
            out = self.down_dim[j](rep_outs[j])
            outs.append(out)  # 64,128,256

        return tuple(outs)  # ([2,64,248,216],[2,128,124,108],[2,256,64,54])
