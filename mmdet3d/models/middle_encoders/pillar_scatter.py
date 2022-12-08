# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from torch import nn

from ..builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.
    将体素特征转换为伪图像
    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.  64
        output_shape (list[int]): Required output shape of features.  [496*432]
    """

    def __init__(self, in_channels, output_shape):  # 64->[496,432]
        super().__init__()
        self.output_shape = output_shape  # [496, 432]
        self.ny = output_shape[0]  # 496
        self.nx = output_shape[1]  # 432
        self.in_channels = in_channels  # 64
        self.fp16_enabled = False

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features. voxel_features:(98,64)  coors:(98,4)  batch_size:2"""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:  # 2
            return self.forward_batch(voxel_features, coors, batch_size)  # (98,64),(98,4),2
        else:
            return self.forward_single(voxel_features, coors)

    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C). [16000,64]
            coors (torch.Tensor): Coordinates of each voxel.体素自身坐标，16000x4，[batch_id, z, y, x]
                The first column indicates the sample ID.第一列表示样本ID
        """
        # Create the canvas for this sample 建立多通道伪图像：通道数=in_channels,img_size=nx*ny=output_shape
        canvas = torch.zeros(
            self.in_channels,  # 64
            self.nx * self.ny,  # 496*432=214272
            dtype=voxel_features.dtype,
            device=voxel_features.device)  # [64,214272]

        # 创建索引 nx*ny
        indices = coors[:, 2] * self.nx + coors[:, 3]  # coors[:, 2]:vy  coors[:, 3]:vx  indices=tensor(一维)
        indices = indices.long()  # 转换成长整形
        voxels = voxel_features.t()  # 将矩阵转置 [64,16000]
        # Now scatter the blob back to the canvas.将原本的voxels特征升维，变为图像大小的维度nx*ny
        canvas[:, indices] = voxels  # [64,496*432]
        # Undo the column stacking to final 4-dim tensor.建立伪图像
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)  # [1,64,432,496]
        return canvas

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).(98,64)
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).(98,4) 4:(batch_id,z,y,x)
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.2
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):  # 2
            # Create the canvas for this sample 创建伪图像画布
            canvas = torch.zeros(
                self.in_channels,  # 64
                self.nx * self.ny,  # 432*496=214272
                dtype=voxel_features.dtype,  # torch.float32
                device=voxel_features.device)  # canvas=(64,214272)

            # Only include non-empty pillars 只包含非空体素
            # 一个值是bool的tensor(98,).对应batch_id的体素坐标为True
            batch_mask = coors[:, 0] == batch_itt  # (True,False)
            # 传入对应batch_id的体素坐标
            this_coors = coors[batch_mask, :]  # (98,4)->(49,4)
            # 编码体素位置索引
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]  # (49,)
            indices = indices.type(torch.long)
            # 传入对应batch_id的体素特征
            voxels = voxel_features[batch_mask, :]  # (98,64)->(49,64)
            voxels = voxels.t()  # (49,64)->(64,49)

            # Now scatter the blob back to the canvas.现在将斑点散射回画布.传入体素特征到对应位置上
            canvas[:, indices] = voxels  # (64,214272)

            # Append to a list for later stacking.附加到列表以供以后堆叠
            batch_canvas.append(canvas)  # [(64,214272),(64,214272)]

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols) 增加tensor维度2->3
        batch_canvas = torch.stack(batch_canvas, 0)  # (2,64,214272)

        # Undo the column stacking to final 4-dim tensor 将tensor最后一维拆分成二维
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)  # (2,64,496,432)

        return batch_canvas  # (2,64,496,432)
