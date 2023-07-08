from functools import partial
from itertools import chain

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmcv.cnn.bricks import build_norm_layer, DropPath
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from ..utils import Scale, lecun_normal_init


class RWKV_TimeMix(nn.Module):

    def __init__(self, n_embd, n_attn, n_head, ctx_len, layer_id=None):
        super().__init__()
        assert n_attn % n_head == 0
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_head = n_head
        self.head_size = n_attn // n_head

        with torch.no_grad(): # initial time_w curves for better convergence
            ww = torch.ones(n_head, ctx_len)
            curve = torch.tensor([-(ctx_len - 1 - i) for i in range(ctx_len)]) # the distance
            for h in range(n_head):
                if h < n_head - 1:
                    decay_speed = math.pow(ctx_len, -(h+1)/(n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)

        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, self.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, self.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(self.ctx_len, 1))      
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)
        self.receptance = nn.Linear(n_embd, n_attn)
        self.output = nn.Linear(n_attn, n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60) # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :]


class RWKV_ChannelMix(nn.Module):

    def __init__(self, n_embd, n_ffn, layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = n_ffn  # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(n_embd, hidden_sz)
        self.value = nn.Linear(n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, n_embd)
        self.receptance = nn.Linear(n_embd, n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v) # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[...,:q.shape[-2],:], sin[...,:q.shape[-2],:]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MHA_rotary(nn.Module):

    def __init__(self, n_embd, n_attn, n_head, ctx_len, layer_id=None, time_shift=False, masked=False):
        super().__init__()
        self.layer_id = layer_id
        assert n_attn % n_head == 0
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.head_size = n_attn // n_head
        self.masked = masked

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.query = nn.Linear(n_embd, n_attn)
        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)

        if masked:
            self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims, base=ctx_len)

        self.output = nn.Linear(n_attn, n_embd)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)    # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        if self.masked:
            att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))  # causal mask
        elif mask is not None:
            att = att.masked_fill(mask[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)  # softmax

        x = att @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)
        return x


class GeGLU(torch.nn.Module):

    def __init__(self, n_embd, n_ffn, layer_id=None, time_shift=False):
        super().__init__()
        self.layer_id = layer_id

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(n_embd, n_ffn)
        self.value = nn.Linear(n_embd, n_ffn)
        self.weight = nn.Linear(n_ffn, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

        k = self.key(x)
        v = self.value(x)        
        y = self.weight(F.gelu(k) * v)
        return y


class MHA_pro(nn.Module):

    def __init__(self, n_embd, n_attn, n_head, ctx_len, layer_id=None, masked=False):
        super().__init__()
        self.layer_id = layer_id
        assert n_attn % n_head == 0
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.head_size = n_attn // n_head
        self.masked = masked

        self.time_w = nn.Parameter(torch.ones(self.n_head, ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))
        if masked:
            self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.query = nn.Linear(n_embd, n_attn)
        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)

        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims, base=ctx_len)

        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False)  # talking heads

        self.output = nn.Linear(n_attn, n_embd)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1) # time-shift mixing
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)    # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        if self.masked:
            att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))  # causal mask
        elif mask is not None:
            att = att.masked_fill(mask[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)  # softmax
        att = att * w  # time-weighting
        att = self.head_mix(att)  # talking heads

        x = att @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x) * self.time_gamma[:T, :]
        return x


class RWKVBlock(nn.Module):

    def __init__(self, n_embd, n_attn, n_head, ctx_len, layer_id=None, model_type='PWKV', mlp_ratio=4,
                 norm_cfg=dict(type='LN', eps=1e-6), drop_path=0., layer_scale_init=None):
        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, n_embd)[1]
        self.norm2 = build_norm_layer(norm_cfg, n_embd)[1]

        n_ffn = int(n_embd * mlp_ratio)
        if model_type == 'RWKV':
            self.attn = RWKV_TimeMix(n_embd, n_attn, n_head, ctx_len, layer_id)
            self.mlp = RWKV_ChannelMix(n_embd, n_ffn=n_ffn, layer_id=layer_id)
        elif model_type == 'MHA_rotary':
            self.attn = MHA_rotary(n_embd, n_attn, n_head, ctx_len, layer_id)
            self.mlp = GeGLU(n_embd, n_ffn=n_ffn, layer_id=layer_id)
        elif model_type == 'MHA_shift':
            self.attn = MHA_rotary(n_embd, n_attn, n_head, ctx_len, layer_id, time_shift=True)
            self.mlp = GeGLU(n_embd, n_ffn=n_ffn, layer_id=layer_id, time_shift=True)
        elif model_type == 'MHA_pro':
            self.attn = MHA_pro(n_embd, n_attn, n_head, ctx_len, layer_id)
            self.mlp = RWKV_ChannelMix(n_embd, n_ffn=n_ffn, layer_id=layer_id)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_init = layer_scale_init is not None
        if self.layer_scale_init:
            self.layer_scale1 = Scale(dim=n_embd, init_value=layer_scale_init)
            self.layer_scale2 = Scale(dim=n_embd, init_value=layer_scale_init)

    def forward(self, x):
        if self.layer_scale_init:
            x = x + self.layer_scale1(self.drop_path1(self.attn(self.norm1(x))))
            x = x + self.layer_scale2(self.drop_path2(self.mlp(self.norm2(x))))
        else:            
            x = x + self.drop_path1(self.attn(self.norm1(x)))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = build_norm_layer(pre_norm, in_channels)[1] if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = build_norm_layer(post_norm, out_channels)[1] if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            B, L, C = x.size()
            H = int(math.sqrt(L))
            x = x.permute(0, 2, 1).view(B, C, H, H)  # [B, L, C] -> [B, C, H, W]
        x = self.conv(x)
        B, C, H, W = x.size()
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, C, H, W] -> [B, L, C]
        x = self.post_norm(x)
        return x


def downsample_layers_four_stages(in_patch_size=7, in_stride=4, in_pad=2,
                                  down_patch_size=3, down_stride=2, down_pad=1):
    r"""
    downsampling (stem) for the first stage is a layer of conv with k7, s4 and p2
    downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
    DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
    use `partial` to specify some arguments
    """
    return [partial(Downsampling,
                kernel_size=in_patch_size, stride=in_stride, padding=in_pad,
                post_norm=dict(type='LN', eps=1e-6), pre_permute=False)] + \
           [partial(Downsampling,
                kernel_size=down_patch_size, stride=down_stride, padding=down_pad,
                pre_norm=dict(type='LN', eps=1e-6), pre_permute=True)] * 3


@BACKBONES.register_module()
class RWKV(BaseBackbone):
    r""" RWKV
        A PyTorch impl of : `RWKV: Reinventing RNNs for the Transformer Era` -
          <https://arxiv.org/abs/2305.13048>`_

        Modified from `the official implementation <https://github.com/BlinkDL/RWKV-LM>`_

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``RWKV.arch_settings``. And if dict, it
            should include the following three keys:

            - depths (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.
            - model_type (list[str]): The type of the token mixer at each stage.
            - head_num (list[int]): Number of attention heads at each stage.
            - mlp_ratio (int): Expand number of channel mixing.

            Defaults to 'rwkv_tiny'.

        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        in_channels (int): Number of input image channels. Defaults to 3.
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage.
            Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init (float or None): Init value for Layer Scale. Default: None.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer.
        out_indices (Sequence[int] or -1): Output from which stages. Default: -1.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
    """
    arch_settings = {
        'rwkv_tiny': {
            'depths': [3, 3, 9, 3],
            'embed_dims': [64, 128, 256, 512],
            'model_type': "RWKV",
            'head_num': [8, 8, 8, 8],
            'mlp_ratio': 4,
        },
        'mha_rotary_tiny': {
            'depths': [3, 3, 9, 3],
            'embed_dims': [64, 128, 256, 512],
            'model_type': "MHA_rotary",
            'head_num': [8, 8, 8, 8],
            'mlp_ratio': 4,
        },
        'mha_shift_tiny': {
            'depths': [3, 3, 9, 3],
            'embed_dims': [64, 128, 256, 512],
            'model_type': "MHA_shift",
            'head_num': [8, 8, 8, 8],
            'mlp_ratio': 4,
        },
        'mha_pro_tiny': {
            'depths': [3, 3, 9, 3],
            'embed_dims': [64, 128, 256, 512],
            'model_type': "MHA_pro",
            'head_num': [8, 8, 8, 8],
            'mlp_ratio': 4,
        },
        'rwkv_small': {
            'depths': [3, 3, 27, 3],
            'embed_dims': [64, 128, 256, 512],
            'model_type': "RWKV",
            'head_num': [8, 8, 8, 8],
            'mlp_ratio': 4,
        },
        'rwkv_base': {
            'depths': [3, 3, 20, 3],
            'embed_dims': [96, 192, 384, 768],
            'model_type': "RWKV",
            'head_num': [8, 8, 8, 8],
            'mlp_ratio': 4,
        },
    }

    def __init__(self,
                 arch='rwkv_small',
                 img_size=224,
                 in_channels=3,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 drop_path_rate=0.,
                 layer_scale_init=1e-6,
                 gap_before_final_norm=True,
                 out_indices=-1,
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs,
                 ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "depths" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        depths = arch['depths']
        embed_dims = arch['embed_dims']
        model_type = arch['model_type']
        head_num = arch['head_num']
        mlp_ratio = arch['mlp_ratio']
        self.num_stage = len(depths)
        self.gap_before_final_norm = gap_before_final_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, list), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stage + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        downsample_layers = downsample_layers_four_stages(in_patch_size, in_stride, in_pad,
                                                          down_patch_size, down_stride, down_pad)
        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * self.num_stage
        down_dims = [in_channels] + embed_dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(self.num_stage)]
        )
        ctx_len = [(img_size // 2**(i+2))**2 for i in range(self.num_stage)]

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList() # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[RWKVBlock(
                    n_embd=embed_dims[i],
                    n_attn=embed_dims[i],
                    n_head=head_num[i],
                    model_type=model_type,
                    ctx_len=ctx_len[i],
                    mlp_ratio=mlp_ratio,
                    norm_cfg=norm_cfg,
                    drop_path=dp_rates[cur + j],
                    layer_scale_init=layer_scale_init,
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = build_norm_layer(norm_cfg, embed_dims[-1])[1]
        self.add_module(f'norm', norm_layer)

        self._freeze_stages()

    def init_weights(self, pretrained=None):
        super(RWKV, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    lecun_normal_init(m, mode='fan_in', distribution='truncated_normal')
                elif isinstance(m, nn.Linear):
                    shape = m.weight.data.shape
                    gain = 1.0  # positive: gain for orthogonal, negative: std for normal
                    scale = 1.0  # extra scale for gain

                    if m.bias is not None:
                        m.bias.data.zero_()
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    if hasattr(m, 'scale_init'):
                        scale = m.scale_init

                    gain *= scale
                    if gain == 0:
                        nn.init.zeros_(m.weight) # zero init is great for some RWKV matrices
                    elif gain > 0:
                        nn.init.orthogonal_(m.weight, gain=gain)
                    else:
                        nn.init.normal_(m.weight, mean=0, std=-gain)
                elif isinstance(m, (
                    nn.LayerNorm, _BatchNorm, nn.GroupNorm)):
                    constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def forward(self, x):
        outs = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                if i == self.num_stage - 1:
                    if self.gap_before_final_norm:
                        x = self.norm(x.mean(dim=1))  # (B, L, C) -> (B, C)
                    else:
                        B, L, C = x.size()
                        H = int(math.sqrt(L))
                        x = self.norm(x).permute(0, 2, 1).view(B, C, H, H)  # [B, L, C] -> [B, C, H, W]
                outs.append(x)

        return outs

    def train(self, mode=True):
        super(RWKV, self).train(mode)
        self._freeze_stages()
