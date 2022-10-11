from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner.base_module import BaseModule

from .attention import CrossMultiheadAttention


class CAETransformerRegressorLayer(BaseModule):
    """Transformer layer for the regressor of CAE.

    This module is different from conventional transformer encoder layer, for
    its queries are the masked tokens, but its keys and values are the
    concatenation of the masked and unmasked tokens.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): The number of heads in multi-head attention.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): The dropout rate. Defaults to 0.0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        init_values (float): The init values of gamma. Defaults to 0.0.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 num_fcs: int = 2,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 init_values: float = 0.0,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 init_cfg: dict = None) -> None:
        super().__init__(init_cfg)

        # NOTE: cross attention
        _, self.norm1_q_cross = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        _, self.norm1_k_cross = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        _, self.norm1_v_cross = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        _, self.norm2_cross = build_norm_layer(norm_cfg, embed_dims, postfix=2)
        self.cross_attn = CrossMultiheadAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=None,
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if init_values > 0:
            self.gamma_1_cross = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
            self.gamma_2_cross = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
        else:
            self.gamma_1_cross = nn.Parameter(
                torch.ones((embed_dims)), requires_grad=False)
            self.gamma_2_cross = nn.Parameter(
                torch.ones((embed_dims)), requires_grad=False)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor,
                pos_q: torch.Tensor, pos_k: torch.Tensor) -> torch.Tensor:
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(
            self.norm1_q_cross(x_q + pos_q),
            k=self.norm1_k_cross(x_kv + pos_k),
            v=self.norm1_v_cross(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.ffn(x))

        return x


class RelativePositionBias(BaseModule):
    """Relative Position Bias.

    This module is copied from
    https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py#L209.

    Args:
        window_size (Sequence[int]): The window size of the relative
            position bias.
        num_heads (int): The number of head in multi-head attention.
    """

    def __init__(self,
                 window_size: Sequence[int],
                 num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each
        # token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] -\
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, ) * 2,
            dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[ 0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[ 0, 0] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self) -> torch.Tensor:
        # WhxWw, WhxWw, nH
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)
        # nH, WhxWw, WhxWw
        return relative_position_bias.permute(2, 0, 1).contiguous()
