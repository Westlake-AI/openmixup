from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import numpy as np

from .pos_embed import apply_rot_embed
from .weight_init import trunc_normal_
from ..helpers import to_2tuple


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class RPEAttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling with relative pos embedding (RPE).
    This is a multi-head attention based replacement for (spatial) average
    pooling in NN architectures.

    Modified from `timm repo
    <https://github.com/rwightman/pytorch-image-models>`
    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: While this impl does not require a fixed feature size, performance at
    differeing resolutions from train varies widely and falls off dramatically.
    I'm not sure if there is a way around this... -RW
    """
    def __init__(self,
                 in_features: int,
                 out_features: int = None,
                 embed_dim: int = None,
                 num_heads: int = 4,
                 qkv_bias: bool = True,
                ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_embed = RPEAttentionPool2d(self.head_dim)

        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        x = x.reshape(B, -1, N).permute(0, 2, 1)

        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)

        x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]

        qc, q = q[:, :, :1], q[:, :, 1:]
        sin_emb, cos_emb = self.pos_embed.get_embed((H, W))
        q = apply_rot_embed(q, sin_emb, cos_emb)
        q = torch.cat([qc, q], dim=2)

        kc, k = k[:, :, :1], k[:, :, 1:]
        k = apply_rot_embed(k, sin_emb, cos_emb)
        k = torch.cat([kc, k], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x[:, 0]


class AttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average
    pooling in NN architectures.

    Modified from `timm repo
    <https://github.com/rwightman/pytorch-image-models>`
    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: This requires feature size upon construction and well prevent adaptive
    sizing of the network.
    """
    def __init__(self,
                 in_features: int,
                 feat_size: Union[int, Tuple[int, int]],
                 out_features: int = None,
                 embed_dim: int = None,
                 num_heads: int = 4,
                 qkv_bias: bool = True,
                ):
        super().__init__()

        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.feat_size = to_2tuple(feat_size)
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        spatial_dim = self.feat_size[0] * self.feat_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(spatial_dim + 1, in_features))
        trunc_normal_(self.pos_embed, std=in_features ** -0.5)
        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        assert self.feat_size[0] == H
        assert self.feat_size[1] == W
        x = x.reshape(B, -1, N).permute(0, 2, 1)
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
        x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x[:, 0]


class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Modified from `timm repo
    <https://github.com/rwightman/pytorch-image-models>`

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports
            3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """
    def __init__(self, channels, filt_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.padding = [get_padding(filt_size, stride, dilation=1)] * 4
        coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (
            self.filt_size - 1)).coeffs.astype(np.float32))
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :].repeat(
            self.channels, 1, 1, 1)
        self.register_buffer('filt', blur_filter, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, 'reflect')
        x = F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)
        return [x]


class MedianPool2d(nn.Module):
    r""" Median pool (usable as median filter when stride=1) module.
    
    Modified from `timm repo
    <https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598>`

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return [x]


class MultiPooling(nn.Module):
    r"""Pooling layers for features from multiple depth.

    Args:
        pool_type (str): Pooling type for the feature map. Options are
            'adaptive' and 'specified'. Defaults to 'adaptive'.
        in_indices (Sequence[int]): Output from which backbone stages.
            Defaults to (0, ).
        backbone (str): The selected backbone. Defaults to 'resnet50'.
    """

    POOL_PARAMS = {
        'resnet50': [
            dict(kernel_size=10, stride=10, padding=4),
            dict(kernel_size=16, stride=8, padding=0),
            dict(kernel_size=13, stride=5, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=6, stride=1, padding=0)
        ]
    }
    POOL_SIZES = {'resnet50': [12, 6, 4, 3, 2]}
    POOL_DIMS = {'resnet50': [9216, 9216, 8192, 9216, 8192]}

    def __init__(self,
                 pool_type='adaptive',
                 in_indices=(0, ),
                 backbone='resnet50'):
        super(MultiPooling, self).__init__()
        assert pool_type in ['adaptive', 'specified']
        if pool_type == 'adaptive':
            self.pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d(self.POOL_SIZES[backbone][l])
                for l in in_indices
            ])
        else:
            self.pools = nn.ModuleList([
                nn.AvgPool2d(**self.POOL_PARAMS[backbone][l])
                for l in in_indices
            ])

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return [p(xx) for p, xx in zip(self.pools, x)]
