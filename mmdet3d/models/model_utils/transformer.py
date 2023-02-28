# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING, MultiheadAttention, FFN
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_attention, build_feedforward_network
from torch import nn as nn
from mmcv.runner import auto_fp16, BaseModule
import torch
import math


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
        self.fp16_enabled = False
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_pos_feats),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    @auto_fp16(out_fp32=True)
    def forward(self, xyz):
        """Forward pass.

        Args:
            xyz (Tensor)： (B, N, 3) the coordinates to embed.

        Returns:
            Tensor: (B, num_pos_feats, N) the embedded position features.
        """
        xyz = xyz.permute(0, 2, 1)
        position_embedding = self.position_embedding_head(xyz)
        position_embedding = position_embedding.permute(0, 2, 1)
        return position_embedding


@ATTENTION.register_module()
class SelfAttention(BaseModule):
    def __init__(self, num_attention_heads, input_size, hidden_size, init_cfg):
        """

        Args:
            num_attention_heads: 多头的头数
            input_size: 输入特征维度
            hidden_size: 输出特征维度
            hidden_dropout_prob: dropout值
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads  # 头数
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 每个头的维度
        self.all_head_size = hidden_size  # 所有头的总维数

        self.query = nn.Conv1d(input_size, self.all_head_size, kernel_size=1, bias=False)  # Wq
        self.key = nn.Conv1d(input_size, self.all_head_size, kernel_size=1, bias=False)  # Wk
        self.value = nn.Conv1d(input_size, self.all_head_size, kernel_size=1, bias=False)  # Wv

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.ffn = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.bn1d = nn.BatchNorm1d(hidden_size)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

        #初始化
        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv1d')

    def transpose_for_scores(self, x):
        """

        将KQV拆分为多个头
        input:mixed_query_layer[batch,n,all_head_size]
        output:query_layer[batch,num_attention_heads,n,attention_head_size]

        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        # 将输入*权重矩阵得到 Q K V
        mixed_query_layer = self.query(input_tensor.transpose(1,2)).transpose(1,2)
        mixed_key_layer = self.key(input_tensor.transpose(1,2)).transpose(1,2)
        mixed_value_layer = self.value(input_tensor.transpose(1,2)).transpose(1,2)

        # 将K Q V拆分为多头
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch,num_attention_heads,n,attention_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [batch,num_attention_heads,n,attention_head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [batch,num_attention_heads,n,attention_head_size]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [batch,num_attention_heads,n,n]
        # 除以向量维度的开方，防止注意力分数随维度增大而增大
        attention_scores = attention_scores / (1e-9 + math.sqrt(self.attention_head_size))  # [batch,num_attention_heads,n,n]

        # Normalize the attention_scores to probabilities.注意力矩阵归一化得到注意力分数
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [batch,num_attention_heads,n,n]
        attention_probs = self.dropout(attention_probs)

        # 注意力分数矩阵*V
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch,num_attention_heads,n,attention_head_size]
        # contiguous()是将tensor的内存变成连续的，为后面的view()做准备
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 将各头的结果拼接起来，减少一个维度
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [batch,n,all_head_size]
        # 应对attention结果做ln,relu
        context_layer = self.bn1d((context_layer + mixed_value_layer).transpose(1,2)).transpose(1,2)

        # 做残差连接的FFN
        hidden_states = self.relu(self.bn1d(self.ffn(context_layer.transpose(1,2))).transpose(1,2))
        hidden_states = self.relu(self.bn1d(self.ffn((hidden_states + context_layer).transpose(1,2))).transpose(1,2))  # 是否俩次都加relu

        return hidden_states  # [batch,n,all_head_size]


@ATTENTION.register_module()
class PillarAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,  # default288
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 pos_encoding=dict(type='ConvBNPositionalEncoding',
                                   input_channel=3,
                                   num_pos_feats=3),
                 **kwargs):
        super().__init__(embed_dims, num_heads, attn_drop, proj_drop,
                         dropout_layer, init_cfg, batch_first, **kwargs)
        self.pos_encoding = build_positional_encoding(pos_encoding)

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
        query_xyz = query[:, :, :3]  # 选取坐标
        key_xyz = key[:, :, :3]
        query_posencode = self.pos_encoding(query_xyz)
        key_posencode = self.pos_encoding(key_xyz)
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

        return attn_out


