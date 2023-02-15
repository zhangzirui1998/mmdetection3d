# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING, MultiheadAttention, FFN
from torch import nn as nn


@ATTENTION.register_module()
class GroupFree3DMHA(MultiheadAttention):
    """A wrapper for torch.nn.MultiheadAttention for GroupFree3D.

    This module implements MultiheadAttention with identity connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension. 编码维度
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`. 多头注意力
        attn_drop (float, optional): A Dropout layer on attn_output_weights.
            Defaults to 0.0. 注意力输出权重的dropout
        proj_drop (float, optional): A Dropout layer. Defaults to 0.0.
        dropout_layer (obj:`ConfigDict`, optional): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`, optional): The Config for
            initialization. Default: None.
        batch_first (bool, optional): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Defaults to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropOut', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(embed_dims, num_heads, attn_drop, proj_drop,
                         dropout_layer, init_cfg, batch_first, **kwargs)

    def forward(self,
                query,
                key,
                value,
                identity,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `GroupFree3DMHA`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                If None, the ``query`` will be used.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link. If None, `x` will be used.
            query_pos (Tensor, optional): The positional encoding for query,
                with the same shape as `x`. Defaults to None.
                If not None, it will be added to `x` before forward function.
            key_pos (Tensor, optional): The positional encoding for `key`,
                with the same shape as `key`. Defaults to None. If not None,
                it will be added to `key` before forward function. If None,
                and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor, optional): ByteTensor mask with shape
                [num_queries, num_keys].
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
            key_padding_mask (Tensor, optional): ByteTensor with shape
                [bs, num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        if hasattr(self, 'operation_name'):
            if self.operation_name == 'self_attn':
                value = value + query_pos
            elif self.operation_name == 'cross_attn':
                value = value + key_pos
            else:
                raise NotImplementedError(
                    f'{self.__class__.name} '
                    f"can't be used as {self.operation_name}")
        else:
            value = value + query_pos

        return super(GroupFree3DMHA, self).forward(
            query=query,
            key=key,
            value=value,
            identity=identity,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)


@POSITIONAL_ENCODING.register_module()
class ConvBNPositionalEncoding(nn.Module):
    """Absolute position embedding with Conv learning.

    Args:
        input_channel (int): input features dim.
        num_pos_feats (int, optional): output position features dim.
            Defaults to 288 to be consistent with seed features dim.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass.

        Args:
            xyz (Tensor)： (B, N, 3) the coordinates to embed.

        Returns:
            Tensor: (B, num_pos_feats, N) the embedded position features.
        """
        xyz = xyz.permute(0, 2, 1)
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@ATTENTION.register_module()
class PillarAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,  # default288
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super().__init__(embed_dims, num_heads, attn_drop, proj_drop,
                         dropout_layer, init_cfg, batch_first, **kwargs)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=True,
                key_pos=True,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        # 判断是否为自注意力
        if key is None:
            key = query
        if value is None:
            value = key
        # 残差结构
        if identity is None:
            identity = query
        # 位置编码
        pos_encoding = ConvBNPositionalEncoding(3, query.shape[-1])
        query_xyz = query[:, :, :3]  # 选取坐标
        key_xyz = key[:, :, :3]
        query_posencode = pos_encoding(query_xyz)
        key_posencode = pos_encoding(key_xyz)
        # key和query使用同一个位置编码
        # if key_pos is None:
        #     if query_pos is not None:
        #         # use query_pos if key_pos is not available
        #         if query_pos.shape == key.shape:
        #             key_pos = query_pos
        #         else:
        #             warnings.warn(f'position encoding of key is'
        #                           f'missing in {self.__class__.__name__}.')
        # key和query使用各自位置编码
        if query_pos:
            if self.batch_first:
                query = query + query_posencode.transpose(1,2) # 特征向量+位置编码
            else:
                query = query + query_posencode
        if key_pos:
            if self.batch_first:
                key = key + key_posencode.transpose(1,2)
            else:
                key = key + key_posencode
        # 交换bs和number位置
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        # 放入attention块中进行计算
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]  # 列表第一个位置是features，第二个位置是weights
        # 恢复bs和number位置
        if self.batch_first:
            out = out.transpose(0, 1)

        attn_out = identity + self.dropout_layer(self.proj_drop(out))  # 残差结构
        # 创建FFN层
        ffn = FFN(embed_dims=attn_out.shape[-1], feedforward_channels=1024, num_fcs=2,ffn_drop=0.,
                  dropout_layer=None, add_identity=True)  # True使用残差
        attn_out = ffn(attn_out)

        return attn_out


