# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class DeltaXYZWLHRBBoxCoder(BaseBBoxCoder):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded. x,y,z,w,l,h,r
    """

    def __init__(self, code_size=7):
        super(DeltaXYZWLHRBBoxCoder, self).__init__()
        self.code_size = code_size  # 7

    @staticmethod
    def encode(src_boxes, dst_boxes):
        """Get box regression transformation deltas (dx, dy, dz, dx_size,
        dy_size, dz_size, dr, dv*) that can be used to transform the
        `src_boxes` into the `target_boxes`.框位置回归：获取预测框与真实框的偏差

        Args:
            src_boxes (torch.Tensor): source boxes, e.g., object proposals.预测框
            dst_boxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.真实框

        Returns:
            torch.Tensor: Box transformation deltas.计算预测框与真实框偏差
        """
        box_ndim = src_boxes.shape[-1]  # 7 对于一维行向量，shape[-1]代表元素总数，即列数
        cas, cgs, cts = [], [], []
        # 列表元素分别赋值到变量
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(
                src_boxes, 1, dim=-1)  # 按照最后一个维度划分，每一份包含一个值：即将src_boxes每个元素依次分配给变量
            xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(
                dst_boxes, 1, dim=-1)
            cts = [g - a for g, a in zip(cgs, cas)]
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(src_boxes, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg = torch.split(dst_boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la**2 + wa**2)  # 对角线长度
        # 定位回归残差
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha  # 在做高度中心回归时考虑了框的高度尺寸变化
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra  # 不做编码角度向量
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dx_size, dy_size,
        dz_size, dr, dv*) to `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, x_size, y_size, z_size, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.根据偏差和预测值计算真实值
        """
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)
