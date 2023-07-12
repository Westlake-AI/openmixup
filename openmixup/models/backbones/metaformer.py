from functools import partial
from itertools import chain

import torch
import torch.nn as nn

from mmcv.cnn.bricks import build_norm_layer, DropPath
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from ..utils import LayerNormGeneral, Scale, lecun_normal_init, to_2tuple


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=0, 
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class SquaredReLU(nn.Module):
    """
    Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True, 
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class IdentityMixing(nn.Identity):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class RandomMixing(nn.Module):

    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


class RandomMixingS4(RandomMixing):

    def __init__(self, num_tokens=49, **kwargs):
        super().__init__(num_tokens=num_tokens)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity, 
                 bias=False, kernel_size=7, padding=3, **kwargs,):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer,
        MetaFormer baslines and related networks.
        Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU,
                 drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixMlp(nn.Module):
    """ MLP with DWConv as used in MetaFormer models, eg Transformer,
        MetaFormer baslines and related networks.
        Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, kernel_size=3, out_features=None, act_layer=StarReLU,
                 drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.hidden_dim = hidden_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.dwconv = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias,
            groups=hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
        norm_layer=dict(type='LN', eps=1e-6), head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = build_norm_layer(norm_layer, hidden_features)[1]
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=dict(type='LayerNormGeneral', eps=1e-6, bias=False),
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None,
                 res_scale_init_value=None):
        super().__init__()
        self.use_bn = norm_layer['type'] == 'BN'

        self.norm1 = build_norm_layer(norm_layer, dim)[1]
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = build_norm_layer(norm_layer, dim)[1]
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        if not self.use_bn:
            x = self.res_scale1(x) + \
                self.layer_scale1(
                    self.drop_path1(self.token_mixer(self.norm1(x)))
                )
            x = self.res_scale2(x) + \
                self.layer_scale2(
                    self.drop_path2(self.mlp(self.norm2(x)))
                )
        else:
            x = self.res_scale1(x) + \
                self.layer_scale1(
                    self.drop_path1(self.token_mixer(
                        self.norm1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)))
                )
            x = self.res_scale2(x) + \
                self.layer_scale2(
                    self.drop_path2(self.mlp(
                        self.norm2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)))
                )
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
                post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6))] + \
           [partial(Downsampling,
                kernel_size=down_patch_size, stride=down_stride, padding=down_pad,
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True)] * 3


@BACKBONES.register_module()
class MetaFormer(BaseBackbone):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision` -
          <https://arxiv.org/abs/2210.13452>`_

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``MetaFormer.arch_settings``. And if dict, it
            should include the following three keys:

            - depths (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.
            - token_mixers (list[str]): The type of the token mixer at each stage.

            Defaults to 'convformer_s18'.

        in_channels (int): Number of input image channels. Defaults to 3.
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        channel_mixers (list, tuple): Mlp for each stage. Default: "Mlp".
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage.
            Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale.
            Default: None. None means not use the layer scale.
            Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale.
            Default: [None, None, 1.0, 1.0]. None means not use the layer scale.
            From: https://arxiv.org/abs/2110.09456.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        out_indices (Sequence[int] or -1): Output from which stages. Default: -1.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
    """
    arch_settings = {
        'identityformer_s12': {
            'depths': [2, 2, 6, 2],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': "IdentityMixing",
        },
        'identityformer_s24': {
            'depths': [4, 4, 12, 4],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': "IdentityMixing",
        },
        'identityformer_s36': {
            'depths': [6, 6, 18, 6],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': "IdentityMixing",
        },
        'identityformer_m36': {
            'depths': [6, 6, 18, 6],
            'embed_dims': [96, 192, 384, 768],
            'token_mixers': "IdentityMixing",
        },
        'identityformer_m48': {
            'depths': [8, 8, 24, 8],
            'embed_dims': [96, 192, 384, 768],
            'token_mixers': "IdentityMixing",
        },
        'randformer_s12': {
            'depths': [2, 2, 6, 2],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': ["IdentityMixing", "IdentityMixing", "RandomMixing", "RandomMixingS4"],
        },
        'randformer_s24': {
            'depths': [4, 4, 12, 4],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': ["IdentityMixing", "IdentityMixing", "RandomMixing", "RandomMixingS4"],
        },
        'randformer_s36': {
            'depths': [6, 6, 18, 6],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': ["IdentityMixing", "IdentityMixing", "RandomMixing", "RandomMixingS4"],
        },
        'randformer_m36': {
            'depths': [6, 6, 18, 6],
            'embed_dims': [96, 192, 384, 768],
            'token_mixers': ["IdentityMixing", "IdentityMixing", "RandomMixing", "RandomMixingS4"],
        },
        'randformer_m48': {
            'depths': [8, 8, 24, 8],
            'embed_dims': [96, 192, 384, 768],
            'token_mixers': ["IdentityMixing", "IdentityMixing", "RandomMixing", "RandomMixingS4"],
        },
        'poolformerv2_s12': {
            'depths': [2, 2, 6, 2],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': "Pooling",
        },
        'poolformerv2_s24': {
            'depths': [4, 4, 12, 4],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': "Pooling",
        },
        'poolformerv2_s36': {
            'depths': [6, 6, 18, 6],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': "Pooling",
        },
        'poolformerv2_m36': {
            'depths': [6, 6, 18, 6],
            'embed_dims': [96, 192, 384, 768],
            'token_mixers': "Pooling",
        },
        'poolformerv2_m48': {
            'depths': [8, 8, 24, 8],
            'embed_dims': [96, 192, 384, 768],
            'token_mixers': "Pooling",
        },
        'convformer_s18': {
            'depths': [3, 3, 9, 3],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': "SepConv",
        },
        'convformer_s36': {
            'depths': [3, 12, 18, 3],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': "SepConv",
        },
        'convformer_m36': {
            'depths': [3, 12, 18, 3],
            'embed_dims': [96, 192, 384, 576],
            'token_mixers': "SepConv",
        },
        'convformer_b36': {
            'depths': [3, 12, 18, 3],
            'embed_dims': [128, 256, 512, 768],
            'token_mixers': "SepConv",
        },
        'caformer_s18': {
            'depths': [3, 3, 9, 3],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': ["SepConv", "SepConv", "Attention", "Attention"],
        },
        'caformer_s36': {
            'depths': [3, 12, 18, 3],
            'embed_dims': [64, 128, 320, 512],
            'token_mixers': ["SepConv", "SepConv", "Attention", "Attention"],
        },
        'caformer_m36': {
            'depths': [3, 12, 18, 3],
            'embed_dims': [96, 192, 384, 576],
            'token_mixers': ["SepConv", "SepConv", "Attention", "Attention"],
        },
        'caformer_b36': {
            'depths': [3, 12, 18, 3],
            'embed_dims': [128, 256, 512, 768],
            'token_mixers': ["SepConv", "SepConv", "Attention", "Attention"],
        },
    }

    def __init__(self,
                 arch='convformer_s18',
                 in_channels=3,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 channel_mixers="Mlp",
                 norm_layers=dict(type='LayerNormGeneral', eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 gap_before_final_norm=True,
                 output_norm=dict(type='LN', eps=1e-6),
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
        token_mixers = arch['token_mixers']
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

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [eval(token_mixers)] * self.num_stage
        else:
            token_mixers = [eval(mixers) for mixers in token_mixers]

        if not isinstance(channel_mixers, (list, tuple)):
            mlps = [eval(channel_mixers)] * self.num_stage
        else:
            mlps = [eval(mixers) for mixers in channel_mixers]

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * self.num_stage

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * self.num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * self.num_stage

        self.stages = nn.ModuleList() # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(
                    dim=embed_dims[i],
                    token_mixer=token_mixers[i],
                    mlp=mlps[i],
                    norm_layer=norm_layers[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_values[i],
                    res_scale_init_value=res_scale_init_values[i],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = build_norm_layer(output_norm, embed_dims[-1])[1]
        self.add_module(f'norm', norm_layer)

        self._freeze_stages()

    def init_weights(self, pretrained=None):
        super(MetaFormer, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    lecun_normal_init(m, mode='fan_in', distribution='truncated_normal')
                elif isinstance(m, nn.Linear):
                    trunc_normal_init(m, mean=0., std=0.02, bias=0)
                elif isinstance(m, (
                    nn.LayerNorm, LayerNormGeneral, _BatchNorm, nn.GroupNorm)):
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
                        x = self.norm(x.mean([1, 2]))  # (B, H, W, C) -> (B, C)
                    else:
                        x = self.norm(x)
                outs.append(x)

        return outs

    def train(self, mode=True):
        super(MetaFormer, self).train(mode)
        self._freeze_stages()
