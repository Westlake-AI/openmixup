import math
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule


class ConditionalPositionEncoding(BaseModule):
    """The Conditional Position Encoding (CPE) module.

    The CPE is the implementation of 'Conditional Positional Encodings
    for Vision Transformers <https://arxiv.org/abs/2102.10882>'_.

    Args:
       in_channels (int): Number of input channels.
       embed_dims (int): The feature dimension. Default: 768.
       stride (int): Stride of conv layer. Default: 1.
    """

    def __init__(self, in_channels, embed_dims=768, stride=1, init_cfg=None):
        super(ConditionalPositionEncoding, self).__init__(init_cfg=init_cfg)
        self.projection = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            groups=embed_dims)
        self.stride = stride

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        feat_token = x
        # convert (B, N, C) to (B, C, H, W)
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.stride == 1:
            x = self.projection(cnn_feat) + cnn_feat
        else:
            x = self.projection(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def resize_relative_position_bias_table(src_shape, dst_shape, table, num_head):
    """Resize relative position bias table.

    Args:
        src_shape (int): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (int): The resolution of downsampled new training
            image, in format (H, W).
        table (tensor): The relative position bias of the pretrained model.
        num_head (int): Number of attention heads.

    Returns:
        torch.Tensor: The resized relative position bias table.
    """
    from scipy import interpolate

    def geometric_progression(a, r, n):
        return a * (1.0 - r**n) / (1.0 - r)

    left, right = 1.01, 1.5
    while right - left > 1e-6:
        q = (left + right) / 2.0
        gp = geometric_progression(1, q, src_shape // 2)
        if gp > dst_shape // 2:
            right = q
        else:
            left = q

    dis = []
    cur = 1
    for i in range(src_shape // 2):
        dis.append(cur)
        cur += q**(i + 1)

    r_ids = [-_ for _ in reversed(dis)]

    x = r_ids + [0] + dis
    y = r_ids + [0] + dis

    t = dst_shape // 2.0
    dx = np.arange(-t, t + 0.1, 1.0)
    dy = np.arange(-t, t + 0.1, 1.0)

    all_rel_pos_bias = []

    for i in range(num_head):
        z = table[:, i].view(src_shape, src_shape).float().numpy()
        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
        all_rel_pos_bias.append(
            torch.Tensor(f_cubic(dx,
                                 dy)).contiguous().view(-1,
                                                        1).to(table.device))
    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

    return new_rel_pos_bias


def build_2d_sincos_position_embedding(patches_resolution,
                                       embed_dims,
                                       temperature=10000.,
                                       cls_token=False):
    """The function is to build position embedding for model to obtain the
    position information of the image patches."""

    if isinstance(patches_resolution, int):
        patches_resolution = (patches_resolution, patches_resolution)

    h, w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dims % 4 == 0, \
        'Embed dimension must be divisible by 4.'
    pos_dim = embed_dims // 4

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])

    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ], dim=1,
    )[None, :, :]

    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb


def pixel_freq_bands(
        num_bands: int,
        max_freq: float = 224.,
        linear_bands: bool = True,
        dtype: torch.dtype = torch.float32,
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=dtype).cuda()
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=dtype).cuda()
    return bands * torch.pi


def inv_freq_bands(
        num_bands: int,
        temperature: float = 100000.,
        step: int = 2,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    inv_freq = 1. / (temperature ** (torch.arange(0, num_bands, step, dtype=dtype).cuda() \
         / num_bands))
    return inv_freq


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    return x * cos_emb + rot(x) * sin_emb


def apply_rot_embed_list(x: List[torch.Tensor], sin_emb, cos_emb):
    if isinstance(x, torch.Tensor):
        x = [x]
    return [t * cos_emb + rot(t) * sin_emb for t in x]


def apply_rot_embed_split(x: torch.Tensor, emb):
    split = emb.shape[-1] // 2
    return x * emb[:, :split] + rot(x) * emb[:, split:]


def build_fourier_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        num_bands: int = 64,
        max_res: int = 224,
        linear_bands: bool = False,
        include_grid: bool = False,
        concat_out: bool = True,
        in_pixels: bool = True,
        dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands, float(max_res), linear_bands=linear_bands, dtype=dtype).cuda()
        else:
            bands = inv_freq_bands(num_bands, step=1, dtype=dtype).cuda()
    else:
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        grid = torch.stack(torch.meshgrid(
            [torch.linspace(-1., 1., steps=s, dtype=dtype).cuda() for s in feat_shape]), dim=-1)
    else:
        grid = torch.stack(torch.meshgrid(
            [torch.arange(s, dtype=dtype).cuda() for s in feat_shape]), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin(), pos.cos()
    out = (grid, pos_sin, pos_cos) if include_grid else (pos_sin, pos_cos)
    # FIXME torchscript doesn't like multiple return types, probably need to always cat?
    if concat_out:
        out = torch.cat(out, dim=-1)
    return out


def build_rotary_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        dim: int = 64,
        max_freq: float = 224,
        linear_bands: bool = False,
        dtype: torch.dtype = torch.float32,
):
    """
    NOTE: shape arg should include spatial dim only
    """
    feat_shape = torch.Size(feat_shape)
    
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape, bands=bands, num_bands=dim // 4, max_res=max_freq,
        linear_bands=linear_bands, concat_out=False, dtype=dtype).cuda()
    N = feat_shape.numel()
    sin_emb = sin_emb.reshape(N, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(N, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


class RotaryEmbed(nn.Module):
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """
    def __init__(self, dim, max_res=224, linear_bands: bool = False):
        super().__init__()
        self.dim = dim
        self.register_buffer('bands', pixel_freq_bands(dim // 4, max_res, linear_bands=linear_bands), persistent=False)

    def get_embed(self, shape: List[int]):
        return build_rotary_pos_embed(shape, self.bands)

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)


class FourierEmbed(nn.Module):

    def __init__(self,
                 max_res: int = 224,
                 num_bands: int = 64,
                 concat_grid=True,
                 keep_spatial=False):
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer('bands', pixel_freq_bands(max_res, num_bands), persistent=False)

    def forward(self, x):
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(
            feat_shape,
            self.bands,
            include_grid=self.concat_grid,
            dtype=x.dtype)
        emb = emb.transpose(-1, -2).flatten(len(feat_shape))
        batch_expand = (B,) + (-1,) * (x.ndim - 1)

        # FIXME support nD
        if self.keep_spatial:
            x = torch.cat(
                [x, emb.unsqueeze(0).expand(batch_expand).permute(0, 3, 1, 2)], dim=1)
        else:
            x = torch.cat(
                [x.permute(0, 2, 3, 1), emb.unsqueeze(0).expand(batch_expand)], dim=-1)
            x = x.reshape(B, feat_shape.numel(), -1)

        return x


class PositionEncodingFourier(BaseModule):
    """The Position Encoding Fourier (PEF) module.

    Modified from `EdgeNeXt <https://arxiv.org/abs/2206.10589>`_

    Args:
        in_channels (int): Number of input channels.
            Default: 32
        embed_dims (int): The feature dimension.
            Default: 768.
        temperature (int): Temperature.
            Default: 10000.
        dtype (torch.dtype): The data type.
            Default: torch.float32.
        init_cfg (dict): The config dict for initializing the module.
            Default: None.
    """

    def __init__(self,
                 in_channels=32,
                 embed_dims=768,
                 temperature=10000,
                 dtype=torch.float32,
                 init_cfg=None):
        super(PositionEncodingFourier, self).__init__(init_cfg=init_cfg)
        self.proj = nn.Conv2d(in_channels * 2, embed_dims, kernel_size=1)
        self.scale = 2 * math.pi
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.dtype = dtype

        dim_t = torch.arange(in_channels, dtype=self.dtype)
        self.dim_t = temperature**(2 * (dim_t // 2) / in_channels)

    def forward(self, bhw_shape):
        B, H, W = bhw_shape
        mask = torch.zeros(B, H, W).bool().to(self.proj.weight.device)
        not_mask = ~mask
        eps = 1e-6
        y_embed = not_mask.cumsum(1, dtype=self.dtype)
        x_embed = not_mask.cumsum(2, dtype=self.dtype)
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.dim_t.to(mask.device)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.proj(pos)

        return pos
