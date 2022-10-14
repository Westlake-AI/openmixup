# Reference: https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/attention.py
from typing import Sequence
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule

from .layer_scale import LayerScale
from ..helpers import to_2tuple


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    A PyTorch implement of : `Swin Transformer: Hierarchical Vision Transformer
    using Shifted Windows <https://arxiv.org/abs/2103.14030>`_

    `attn_scale` is modified from : `Anti-Oversmoothing in Deep Vision
    Transformers via the Fourier Domain Analysis: From Theory to Practice
    <https://arxiv.org/abs/2203.05962>`_

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_scale=False,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_scale = attn_scale
        if self.attn_scale:
            self.lamb = nn.Parameter(
                torch.zeros(num_heads), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        super(WindowMSA, self).init_weights()

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        if self.attn_scale:
            attn_d = torch.ones(
                attn.shape[-2:], device=attn.device) / N  # [l, l]
            attn_d = attn_d[None, None, ...]  # [B, N, l, l]
            attn_h = attn - attn_d  # [B, N, l, l]
            attn_h = attn_h * (1. + self.lamb[None, :, None, None]
                               )  # [B, N, l, l]
            attn = attn_d + attn_h  # [B, N, l, l]

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shift Window Multihead Self-Attention Module.

    A PyTorch implement of : `Swin Transformer: Hierarchical Vision Transformer
    using Shifted Windows <https://arxiv.org/abs/2103.14030>`_

    `attn_scale` is modified from : `Anti-Oversmoothing in Deep Vision
    Transformers via the Fourier Domain Analysis: From Theory to Practice
    <https://arxiv.org/abs/2203.05962>`_

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults to dict(type='DropPath', drop_prob=0.).
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0,
                 attn_scale=False,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 pad_small_map=False,
                 input_resolution=None,
                 auto_pad=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        if input_resolution is not None or auto_pad is not None:
            warnings.warn(
                'The ShiftWindowMSA in new version has supported auto padding '
                'and dynamic input shape in all condition. And the argument '
                '`auto_pad` and `input_resolution` have been deprecated.',
                DeprecationWarning)

        self.shift_size = shift_size
        self.window_size = window_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_scale=attn_scale,
        )

        self.drop = build_dropout(dropout_layer)
        self.pad_small_map = pad_small_map

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, f"The query length {L} doesn't match the input "\
            f'shape ({H}, {W}).'
        query = query.view(B, H, W, C)

        window_size = self.window_size
        shift_size = self.shift_size

        if min(H, W) == window_size:
            # If not pad small feature map, avoid shifting when the window size
            # is equal to the size of feature map. It's to align with the
            # behavior of the original implementation.
            shift_size = shift_size if self.pad_small_map else 0
        elif min(H, W) < window_size:
            # In the original implementation, the window size will be shrunk
            # to the size of feature map. The behavior is different with
            # swin-transformer for downstream tasks. To support dynamic input
            # shape, we don't allow this feature.
            assert self.pad_small_map, \
                f'The input shape ({H}, {W}) is smaller than the window ' \
                f'size ({window_size}). Please set `pad_small_map=True`, or ' \
                'decrease the `window_size`.'

        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))

        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if shift_size > 0:
            query = torch.roll(
                query, shifts=(-shift_size, -shift_size), dims=(1, 2))

        attn_mask = self.get_attn_mask((H_pad, W_pad),
                                       window_size=window_size,
                                       shift_size=shift_size,
                                       device=query.device)

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(query, window_size)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad,
                                        window_size)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if H != H_pad or W != W_pad:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)

        return x

    @staticmethod
    def window_reverse(windows, H, W, window_size):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    @staticmethod
    def get_attn_mask(hw_shape, window_size, shift_size, device=None):
        if shift_size > 0:
            img_mask = torch.zeros(1, *hw_shape, 1, device=device)
            h_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = ShiftWindowMSA.window_partition(
                img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        return attn_mask


class MultiheadAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    `attn_scale` is modified from : `Anti-Oversmoothing in Deep Vision
    Transformers via the Fourier Domain Analysis: From Theory to Practice
    <https://arxiv.org/abs/2203.05962>`_

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Whether to use the layer init scale. Defaults
            to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 attn_scale=False,
                 v_shortcut=False,
                 use_layer_scale=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_scale = attn_scale
        if self.attn_scale:
            self.lamb = nn.Parameter(
                torch.zeros(num_heads), requires_grad=True)

        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

        if use_layer_scale:
            self.gamma1 = LayerScale(embed_dims, data_format='channels_last')
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if self.attn_scale:
            attn_d = torch.ones(
                attn.shape[-2:], device=attn.device) / N  # [l, l]
            attn_d = attn_d[None, None, ...]  # [B, N, l, l]
            attn_h = attn - attn_d  # [B, N, l, l]
            attn_h = attn_h * (1. + self.lamb[None, :, None, None]
                               )  # [B, N, l, l]
            attn = attn_d + attn_h  # [B, N, l, l]

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class MultiheadAttentionWithRPE(MultiheadAttention):
    """Multi-head Attention Module with relative position.

    This module rewrite the MultiheadAttention in MMSelfSup by adding the
    relative position bias.

    `attn_scale` is modified from : `Anti-Oversmoothing in Deep Vision
    Transformers via the Fourier Domain Analysis: From Theory to Practice
    <https://arxiv.org/abs/2203.05962>`_

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        window_size (int): The window size of the relative position bias.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 window_size: int,
                 input_dims: int = None,
                 attn_drop: float = 0,
                 proj_drop: float = 0,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 proj_bias: bool = True,
                 attn_scale: bool = False,
                 init_cfg: dict = None) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            input_dims=input_dims,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias,
            attn_scale=attn_scale,
            init_cfg=init_cfg)

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        assert isinstance(window_size, Sequence)
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        # relative_position_bias_table shape is (2*Wh-1 * 2*Ww-1 + 3, nH)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for
        # each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # coords shape is (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # coords_flatten shape is (2, Wh*Ww)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :])
        # relative_coords shape is (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, ) * 2,
            dtype=relative_coords.dtype)

        # relative_position_index shape is (Wh*Ww, Wh*Ww)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index',
                             relative_position_index)

        self.attn_scale = attn_scale
        if self.attn_scale:
            self.lamb = nn.Parameter(
                torch.zeros(num_heads), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        B, N, _ = x.shape
        qkv = F.linear(
            x, weight=self.qkv.weight,
            bias=qkv_bias).reshape(B, N, 3, self.num_heads,
                                   self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[
                    self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)

        if self.attn_scale:
            attn_d = torch.ones(
                attn.shape[-2:], device=attn.device) / N  # [l, l]
            attn_d = attn_d[None, None, ...]  # [B, N, l, l]
            attn_h = attn - attn_d  # [B, N, l, l]
            attn_h = attn_h * (1. + self.lamb[None, :, None, None]
                               )  # [B, N, l, l]
            attn = attn_d + attn_h  # [B, N, l, l]

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def attention_pool(tensor, pool, hw_shape, has_cls_embed=True, norm=None):
    """ Spatial pooling for MultiheadPoolAttention """
    if pool is None:
        return tensor, hw_shape
    if tensor.ndim == 4:
        pass
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, _, C = tensor.shape
    H, W = hw_shape
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()
    tensor = pool(tensor)

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)

    if tensor.ndim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape


def cal_rel_pos_spatial(
        attn, q, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w,):
    """ Spatial Relative Positional Embeddings """
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)
    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class MultiheadPoolAttention(BaseModule):
    """Multi-head Pooling Attention Module.

    Modified from `Improved Multiscale Vision Transformers for Classification
    and Detection <https://arxiv.org/abs/2112.01526>`_

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        input_size (int, optional): The input size of feature maps.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        spatial_mode (str): Mode of spatial mixing in {'conv', 'conv_unshared',
            'avgpool', 'maxpool'}. Defaults to 'conv'.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        pool_first (bool): Whether to pool before spatial attention.
        pool_residual (bool): Whether to add residual path for pooling.
        rel_pos_spatial (bool): Whether to use relative pos_embed.
        rel_pos_zero_init (bool): Whether to init the pos_embed as zero.
        with_cls_token (bool): Whether to use cls_token (cls embedding).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 input_dims=None,
                 input_size=None,
                 num_heads=8,
                 attn_drop=0.,
                 proj_drop=0.,
                 spatial_mode='conv',
                 kernel_q=1,
                 kernel_kv=1,
                 stride_q=1,
                 stride_kv=1,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 qkv_bias=True,
                 proj_bias=True,
                 pool_first=False,
                 pool_residual=True,
                 rel_pos_spatial=False,
                 rel_pos_zero_init=False,
                 with_cls_token=True,
                 init_cfg=None):
        super(MultiheadPoolAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.spatial_mode = spatial_mode
        self.pool_first = pool_first
        self.pool_residual = pool_residual
        self.with_cls_token = with_cls_token
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5
        assert spatial_mode in ['conv', 'conv_unshared', 'avgpool', 'maxpool',]

        # Skip pooling with kernel and stride size of (1, 1, 1).
        kernel_q = _pair(kernel_q) if isinstance(kernel_q, int) else kernel_q
        kernel_kv = _pair(kernel_kv) if isinstance(kernel_kv, int) else kernel_kv
        stride_q = _pair(stride_q) if isinstance(stride_q, int) else stride_q
        stride_kv = _pair(stride_kv) if isinstance(stride_kv, int) else stride_kv
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q, stride_q = None, None
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv, stride_kv = None, None
        padding_q = _pair(int(kernel_q[0] // 2))
        padding_kv = _pair(int(kernel_kv[0] // 2))

        if pool_first:
            self.q = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
            self.k = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
            self.v = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.spatial_mode in ['avgpool', 'maxpool']:
            pool_op = nn.MaxPool2d if self.spatial_mode == "maxpool" else nn.AvgPool2d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if kernel_q is not None else None)
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if kernel_kv is not None else None)
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if kernel_kv is not None else None)
        elif self.spatial_mode in ['conv', 'conv_unshared']:
            if pool_first:
                dim_conv = input_dims // num_heads \
                    if self.spatial_mode == "conv" else input_dims
            else:
                dim_conv = embed_dims // num_heads \
                    if self.spatial_mode == "conv" else embed_dims
            self.pool_q = (
                nn.Conv2d(dim_conv, dim_conv, kernel_q, stride=stride_q,
                          padding=padding_q, groups=dim_conv, bias=False) \
                if kernel_q is not None else None
            )
            self.norm_q = build_norm_layer(norm_cfg, dim_conv)[1] \
                if kernel_q is not None else None
            self.pool_k = (
                nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv,
                          padding=padding_kv, groups=dim_conv, bias=False) \
                if kernel_kv is not None else None
            )
            self.norm_k = build_norm_layer(norm_cfg, dim_conv)[1] \
                if kernel_kv is not None else None
            self.pool_v = (
                nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv,
                          padding=padding_kv, groups=dim_conv, bias=False) \
                if kernel_kv is not None else None
            )
            self.norm_v = build_norm_layer(norm_cfg, dim_conv)[1] \
                if kernel_kv is not None else None

        # relative pos embedding
        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            if isinstance(input_size, int):
                input_size = to_2tuple(input_size)
            else:
                assert input_size[0] == input_size[1]
            q_size = input_size[0] // stride_q[1] \
                if stride_q is not None else input_size[0]
            kv_size = input_size[0] // stride_kv[1] \
                if stride_kv is not None else input_size[0]
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dims))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dims))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, hw_shape):
        B, N, _ = x.shape
        if self.pool_first:
            fold_dim = 1 if self.spatial_mode == "conv_unshared" else self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.spatial_mode != "conv_unshared"
            qkv = self.qkv(x).reshape(B, N, 3,
                                      self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_shape = attention_pool(
            q, self.pool_q, hw_shape, has_cls_embed=self.with_cls_token,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k, self.pool_k, hw_shape, has_cls_embed=self.with_cls_token,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v, self.pool_v, hw_shape, has_cls_embed=self.with_cls_token,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:
            q_N = np.prod(q_shape) + 1 if self.with_cls_token else np.prod(q_shape)
            k_N = np.prod(k_shape) + 1 if self.with_cls_token else np.prod(k_shape)
            v_N = np.prod(v_shape) + 1 if self.with_cls_token else np.prod(v_shape)

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn, q, self.with_cls_token,
                q_shape, k_shape, self.rel_pos_h, self.rel_pos_w,
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        if self.pool_residual:
            x = x + q
        x = x.transpose(1, 2).reshape(B, -1, self.embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class HiLoAttention(BaseModule):
    """HiLo Attention Module.

    Modified from `Fast Vision Transformers with HiLo Attention
    <https://arxiv.org/abs/2205.13213>`_

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        window_size (int): The local window size of HiLo attention.
        alpha (float): Ratio to split the attention to high and low parts.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 window_size=2,
                 alpha=0.5,
                 input_dims=None,
                 attn_drop=0,
                 proj_drop=0,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 init_cfg=None):
        super(HiLoAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        assert embed_dims % num_heads == 0, \
            f"embed_dims {embed_dims} should be divided by num_heads {num_heads}."
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
        self.window_size = window_size

        # self-attention heads in Lo-Fi and Lo-Fi
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * self.head_dims
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * self.head_dims

        if self.window_size == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = embed_dims

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.window_size != 1:
                self.pool = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.input_dims, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.input_dims, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim, bias=proj_bias)

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.input_dims, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim, bias=proj_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.window_size, W // self.window_size
        total_groups = h_group * w_group

        x = x.reshape(
            B, h_group, self.window_size, w_group, self.window_size, C).transpose(2, 3)
        qkv = self.h_qkv(x).reshape(
            B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(
            B, h_group, w_group, self.window_size, self.window_size, self.h_dim)
        x = x.transpose(2, 3).reshape(
            B, h_group * self.window_size, w_group * self.window_size, self.h_dim)

        x = self.h_proj(x)
        x = self.proj_drop(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(
            B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.window_size > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.pool(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(
                B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(
                B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)

        x = self.l_proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C)

        if self.h_heads == 0:
            x = self.lofi(x).reshape(B, N, C)
            return self.out_drop(x)

        if self.l_heads == 0:
            x = self.hifi(x).reshape(B, N, C)
            return self.out_drop(x)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)
        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, C)

        x = self.out_drop(x)
        return x


class FlowAttention(BaseModule):
    """Multi-head Attention Module.

    Modified from `Flowformer: Linearizing Transformers with Conservation
    Flows <https://arxiv.org/abs/2202.06258>`_

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=True,
                 proj_bias=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def kernel(self, x):
        """ non-neg of the in/output flow """
        x = torch.sigmoid(x)
        return x

    def my_sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # kernel for non-neg
        q, k = self.kernel(q), self.kernel(k)
        # normalizer
        sink_incoming = 1.0 / (
            self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (
            self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        conserved_sink = self.my_sum(
            q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(
            k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(
            conserved_source, min=-1.0, max=1.0)  # for stability

        # allocation
        sink_allocation = torch.sigmoid(
            conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        sink_allocation = self.attn_drop(sink_allocation)
        # competition
        source_competition = torch.softmax(
            conserved_source, dim=-1) * float(k.shape[2])
        source_competition = self.attn_drop(source_competition)
        # multiply
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * \
            sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        x = (x_update).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossMultiheadAttention(BaseModule):
    """Cross attention between queries and the union of keys and values.

    This module is different from ``MultiheadAttention``, for the attention
    is computed between queries and the union of keys and values.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(embed_dims, embed_dims, bias=False)
        self.k = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v = nn.Linear(embed_dims, embed_dims, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                x: torch.Tensor,
                k: torch.Tensor = None,
                v: torch.Tensor = None) -> None:
        B, N, _ = x.shape

        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(
            input=x, weight=self.q.weight, bias=q_bias)  # (B, N_q, dim)
        k = F.linear(
            input=k, weight=self.k.weight, bias=k_bias)  # (B, N_k, dim)
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)

        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape(B, N_k, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape(B, N_v, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ChannelMultiheadAttention(BaseModule):
    """Channel Multihead Self-attention Module.

    This module implements channel multi-head attention that supports different
    input dims and embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shoutcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to False.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        qk_scale_type (str): The scale type of qk scale.
            Defaults to 'learnable'. It can be 'learnable', 'fixed' or 'none'.
        qk_scale (float, optional): If set qk_scale_type to 'none', this
            should be specified with valid float number. Defaults to None.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=False,
                 proj_bias=True,
                 qk_scale_type='learnable',
                 qk_scale=None,
                 v_shortcut=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        if qk_scale_type == 'learnable':
            self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        elif qk_scale_type == 'fixed':
            self.scale = self.head_dims**-0.5
        elif qk_scale_type == 'none':
            assert qk_scale is not None
            self.scale = qk_scale

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)

        q, k, v = [item.transpose(-2, -1) for item in [qkv[0], qkv[1], qkv[2]]]

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = qkv[2].squeeze(1) + x
        return x
