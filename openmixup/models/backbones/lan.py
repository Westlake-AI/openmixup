import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init

from ..utils import channel_shuffle
from ..registry import BACKBONES
from .base_backbone import BaseBackbone


def custom_build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    if cfg is None:
        return nn.Identity()
    if cfg['type'] == 'SiLU':
        return nn.SiLU()
    else:
        return build_activation_layer(cfg)


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims=128, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class MixFFN(BaseModule):
    """An implementation of MixFFN of VAN.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Depth-wise Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg

        self.fc1 = Conv2d(
            in_channels=embed_dims,
            out_channels=feedforward_channels,
            kernel_size=1)
        self.dwconv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=feedforward_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvFFN(BaseModule):
    """An implementation of Conv FFN

    Args:
        in_features (int): The feature dimension.
        hidden_features (int): The hidden dimension of FFNs.
        out_features (int): The output dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels=None,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(ConvFFN, self).__init__(init_cfg=init_cfg)

        feedforward_channels = feedforward_channels or embed_dims
        self.fc1 = Conv2d(embed_dims, feedforward_channels, 1)
        self.fc2 = Conv2d(feedforward_channels, embed_dims, 1)
        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DecomposeFFN(BaseModule):
    """An implementation of FFN with Feature Decomposing.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 decompose_method='after',
                 decompose_init_value=0.,
                 decompose_act_cfg=None,
                 decompose_post_conv=False,
                 init_cfg=None):
        super(DecomposeFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        assert decompose_post_conv == False

        self.fc1 = Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        assert decompose_method in [None, 'between', 'between-shortcut', 'after',]
        self.decompose_method = decompose_method
        self.decompose = Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        ) if decompose_method is not None else nn.Identity()
        self.sigma = ElementScale(
            self.feedforward_channels, decompose_init_value, requires_grad=True)
        self.decompose_act = custom_build_activation_layer(decompose_act_cfg) \
            if decompose_method is not None else nn.Identity()

    def feat_decompose(self, x, shortcut=None):
        x_d = shortcut if shortcut is not None else x
        x_d = self.decompose_act(self.decompose(x))  # [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - x_d)
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        if self.decompose_method == 'between-shortcut':
            x = self.feat_decompose(self.dwconv(x), shortcut=x)
        else:
            x = self.dwconv(x)
        if self.decompose_method == 'between':
            x = self.feat_decompose(x)
        x = self.act(x)
        x = self.drop(x)
        if self.decompose_method == 'after':
            x = self.feat_decompose(x)
        # proj 2
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(BaseModule):
    """Large Kernel Attention(LKA) of VAN.

    .. code:: text
            DW_conv (depth-wise convolution)
                            |
                            |
        DW_D_conv (depth-wise dilation convolution)
                            |
                            |
        Transition Convolution (1×1 convolution)

    Args:
        embed_dims (int): Number of input channels.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_kernel_size=5,
                 with_dilation=True,
                 with_pointwise=True,
                 init_cfg=None):
        super(LKA, self).__init__(init_cfg=init_cfg)

        # a spatial local convolution (depth-wise convolution)
        self.DW_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2,
            groups=embed_dims)
        # a spatial long-range convolution (depth-wise dilation convolution)
        self.DW_D_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=embed_dims,
            dilation=3) if with_dilation else nn.Identity()
        # a channel convolution (point-wise convolution)
        self.PW_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1) if with_pointwise else nn.Identity()

    def forward(self, x):
        u = x.clone()
        attn = self.DW_conv(x)
        attn = self.DW_D_conv(attn)
        attn = self.PW_conv(attn)

        return u * attn


class VANAttention(BaseModule):
    """Local attention module in VANBloack.

    Args:
        embed_dims (int): Number of input channels.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_kernel_size=5,
                 act_cfg=dict(type='GELU'),
                 with_dilation=True,
                 with_pointwise=True,
                 init_cfg=None):
        super(VANAttention, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.proj_1 = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = LKA(
            embed_dims, dw_kernel_size,
            with_dilation=with_dilation, with_pointwise=with_pointwise)
        self.proj_2 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKGAU(BaseModule):
    """Gated Attention Unit with Large Kernel (LKGAU).

    Args:
        embed_dims (int): Number of input channels.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_kernel_size=5,
                 with_dilation=True,
                 with_pointwise=True,
                 init_cfg=None):
        super(LKGAU, self).__init__(init_cfg=init_cfg)

        # a spatial local convolution (depth-wise convolution)
        self.DW_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2,
            groups=embed_dims)
        # a spatial long-range convolution (depth-wise dilation convolution)
        self.DW_D_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=embed_dims,
            dilation=3) if with_dilation else nn.Identity()
        # a channel convolution
        self.PW_conv = Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1) if with_pointwise else nn.Identity()

    def forward(self, x):
        x = self.DW_conv(x)
        x = self.DW_D_conv(x)
        x = self.PW_conv(x)
        return x


class InceptionGAU(BaseModule):
    """Gated Attention Unit with Inception Kernel (InceptionGAU).

    Args:
        embed_dims (int): Number of input channels.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_kernel_size=5,
                 with_channel_split=[2, 1, 1,],
                 with_dilation=True,
                 with_pointwise=True,
                 init_cfg=None):
        super(InceptionGAU, self).__init__(init_cfg=init_cfg)

        assert len(with_channel_split) == 3
        self.split_ratio = [i / sum(with_channel_split) for i in with_channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims

        assert with_dilation == True and with_pointwise == True
        assert dw_kernel_size % 2 == 1 and dw_kernel_size >= 3
        # basic DW conv
        self.DW_conv0 = Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2,
            groups=self.embed_dims)
        # DW conv 1
        self.DW_conv1 = Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5 if dw_kernel_size != 7 else 7,
            stride=1,
            padding=4 if dw_kernel_size != 7 else 6,
            groups=self.embed_dims_1,
            dilation=2,
        )
        # DW conv 2
        self.DW_conv2 = Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=self.embed_dims_2,
            dilation=3,
        )
        # a channel convolution
        self.PW_conv = Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...],
            x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class GAUAttention(BaseModule):
    """Local attention module in VANBloack.

    Args:
        embed_dims (int): Number of input channels.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_kernel_size=5,
                 act_value_kernel=dict(type="GELU"),
                 act_gate_kernel=dict(type="SiLU"),
                 with_dilation=True,
                 with_pointwise=True,
                 with_channel_shuffle=False,
                 init_cfg=None):
        super(GAUAttention, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.with_channel_shuffle = with_channel_shuffle
        self.proj_1 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.proj_g = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        # value
        self.large_kernel_unit = LKGAU(
            embed_dims, dw_kernel_size,
            with_dilation=with_dilation,
            with_pointwise=with_pointwise,
        )
        self.proj_2 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = custom_build_activation_layer(act_value_kernel)
        self.act_gate = custom_build_activation_layer(act_gate_kernel)

    def forward(self, x):
        shorcut = x.clone()
        x = self.act_value(self.proj_1(x))

        # gating * value
        v = self.large_kernel_unit(x)
        g = self.proj_g(x)
        x = self.act_gate(g) * self.act_value(v)

        if self.with_channel_shuffle:
            x = channel_shuffle(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class InceptionGAUAttention(BaseModule):
    """Local attention module in VANBloack.

    Args:
        embed_dims (int): Number of input channels.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_kernel_size=5,
                 act_value_kernel=dict(type="GELU"),
                 act_gate_kernel=dict(type="SiLU"),
                 with_channel_split=[2, 1, 1],
                 with_dilation=True,
                 with_pointwise=True,
                 with_channel_shuffle=False,
                 decompose_method=None,
                 decompose_position='before',
                 init_cfg=None):
        super(InceptionGAUAttention, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.proj_1 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        # value
        self.value = InceptionGAU(
            embed_dims, dw_kernel_size,
            with_channel_split=with_channel_split,
            with_dilation=with_dilation,
            with_pointwise=with_pointwise,
        )
        self.proj_2 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.channel_split_group = sum(with_channel_split)
        assert embed_dims % self.channel_split_group == 0
        assert with_channel_shuffle == False

        # activation for gating and value
        self.act_value = custom_build_activation_layer(act_value_kernel)
        self.act_gate = custom_build_activation_layer(act_gate_kernel)

        # decompose
        self.decompose_position = decompose_position if decompose_method is not None else 'none'
        assert decompose_method in [None, 'pool',]
        assert decompose_position in ['before', 'between', 'between-shortcut', 'after',]
        if decompose_method is not None:
            self.sigma = ElementScale(embed_dims, 0., requires_grad=True)
        else:
            self.sigma = None

    def feat_decompose(self, x, shortcut=None):
        x_d = shortcut if shortcut is not None else x
        x_d = F.adaptive_avg_pool2d(x_d, output_size=1)  # [B, C, 1, 1]
        x = x + self.sigma(x - x_d)
        return x

    def forward(self, x):
        shortcut = x.clone()

        if self.decompose_position == 'before':
            x = self.feat_decompose(x)
        x = self.proj_1(x)
        if self.decompose_position == 'between':
            x = self.feat_decompose(x)
        if self.decompose_position == 'between-shortcut':
            x = self.feat_decompose(x, shortcut=shortcut)
        x = self.act_value(x)
        if self.decompose_position == 'after':
            x = self.feat_decompose(x)

        # gating * value
        x = self.act_gate(self.gate(x)) * self.act_value(self.value(x))
        x = self.proj_2(x)
        x = x + shortcut
        return x


class VANBlock(BaseModule):
    """A block of VAN.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-2.
        attention_types (str): Type of attention in each stage.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN', eps=1e-5),
                 layer_scale_init_value=1e-2,
                 attention_types=None,
                 ffn_types=None,
                 with_channel_split=[2, 1, 1,],
                 attn_act_gate_cfg=dict(type='GELU'),
                 attn_act_value_cfg=dict(type='GELU'),
                 attn_dw_kernel_size=5,
                 attn_with_dilation=True,
                 attn_with_pointwise=True,
                 attn_with_channel_shuffle=False,
                 attn_decompose_method=None,
                 attn_decompose_position='before',
                 ffn_dwconv_kernel_size=3,
                 ffn_decompose_method='after',
                 ffn_decompose_init_value=1,
                 ffn_decompose_act_cfg=None,
                 ffn_decompose_post_conv=False,
                 init_cfg=None):
        super(VANBlock, self).__init__(init_cfg=init_cfg)
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        # attention
        if attention_types == "GAU":
            self.attn = GAUAttention(
                embed_dims,
                dw_kernel_size=attn_dw_kernel_size,
                act_value_kernel=attn_act_value_cfg,
                act_gate_kernel=attn_act_gate_cfg,
                with_dilation=attn_with_dilation,
                with_pointwise=attn_with_pointwise,
                with_channel_shuffle=attn_with_channel_shuffle,
            )
        elif attention_types == "InceptionGAU":
            self.attn = InceptionGAUAttention(
                embed_dims,
                dw_kernel_size=attn_dw_kernel_size,
                act_value_kernel=attn_act_value_cfg,
                act_gate_kernel=attn_act_gate_cfg,
                with_channel_split=with_channel_split,
                with_dilation=attn_with_dilation,
                with_pointwise=attn_with_pointwise,
                with_channel_shuffle=attn_with_channel_shuffle,
                decompose_method=attn_decompose_method,
                decompose_position=attn_decompose_position,
            )
        else:
            self.attn = VANAttention(
                embed_dims,
                act_cfg=attn_act_value_cfg,
                dw_kernel_size=attn_dw_kernel_size,
                with_dilation=attn_with_dilation,
                with_pointwise=attn_with_pointwise)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        # feed forward MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        if ffn_types == "Mix":
            self.mlp = MixFFN(  # dwconv + FFN
                embed_dims=embed_dims,
                feedforward_channels=mlp_hidden_dim,
                act_cfg=act_cfg,
                kernel_size=ffn_dwconv_kernel_size,
                ffn_drop=drop_rate)
        elif ffn_types == "Decompose":
            self.mlp = DecomposeFFN(  # Decomposed FFN
                embed_dims=embed_dims,
                feedforward_channels=mlp_hidden_dim,
                act_cfg=act_cfg,
                kernel_size=ffn_dwconv_kernel_size,
                ffn_drop=drop_rate,
                decompose_method=ffn_decompose_method,
                decompose_init_value=ffn_decompose_init_value,
                decompose_act_cfg=ffn_decompose_act_cfg,
                decompose_post_conv=ffn_decompose_post_conv,
            )
        else:
            self.mlp = ConvFFN(  # vanilla FFN
                embed_dims=embed_dims,
                feedforward_channels=mlp_hidden_dim,
                act_cfg=act_cfg,
                ffn_drop=drop_rate)

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        if self.layer_scale_1 is not None:
            x = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x
        x = identity + self.drop_path(x)

        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.layer_scale_2 is not None:
            x = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x
        x = identity + self.drop_path(x)

        return x


class VANPatchEmbed(PatchEmbed):
    """Image to Patch Embedding of VAN.

    The differences between VANPatchEmbed & PatchEmbed:
        1. Use BN.
        2. Do not use 'flatten' and 'transpose'.
    """

    def __init__(self, *args, norm_cfg=dict(type='BN'), **kwargs):
        super(VANPatchEmbed, self).__init__(*args, norm_cfg=norm_cfg, **kwargs)

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
            tuple: Contains merged results and its spatial shape.
            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class MiddleEmbedding(BaseModule):
    """An implementation of Conv middle embedding layer.

    Args:
        in_features (int): The feature dimension.
        out_features (int): The output dimension of FFNs.
        kernel_size (int): The conv kernel size of middle patch embedding.
            Defaults to 3.
        stride_size (int): The conv stride of middle patch embedding.
            Defaults to 2.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
    """

    def __init__(self,
                 in_channels,
                 embed_dims=None,
                 kernel_size=3,
                 stride_size=2,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None,
                ):
        super(MiddleEmbedding, self).__init__(init_cfg)

        embed_dims = in_channels or embed_dims
        self.projection = nn.Sequential(
            Conv2d(  # point-wise conv
                in_channels, embed_dims, kernel_size=1,
                stride=1, padding=0),
            Conv2d(  # depth-wise conv
                in_channels, embed_dims, kernel_size=kernel_size,
                stride=stride_size, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, embed_dims)[1],
        )

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


@BACKBONES.register_module()
class LAN(BaseBackbone):
    """Linear Attention Network based on Visual Attention Network.
        v08.17, IP53

    Args:
        arch (str | dict): Visual Attention Network architecture.
            If use string, choose from 'tiny', 'small', 'base' and 'large'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **ffn_ratios** (List[int]): The number of expansion ratio of
            feedforward network hidden layer channels.

            Defaults to 'tiny'.
        patch_sizes (List[int | tuple]): The patch size in patch embeddings.
            Defaults to [7, 3, 3, 3].
        in_channels (int): The num of input channels. Defaults to 3.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        init_value (float): Init value for Layer Scale. Defaults to 1e-2.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to ``dict(type='LN')``.
        conv_norm_cfg (dict): Config dict for convolution normalization layer.
            Defaults to ``dict(type='BN')``.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from openmixup.models import VAN
        >>> import torch
        >>> cfg = dict(arch='tiny')
        >>> model = VAN(**cfg)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> outputs = model(inputs)
        >>> for out in outputs:
        >>>     print(out.size())
        (1, 256, 7, 7)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 160, 256],
                         'depths': [3, 3, 5, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 2, 4, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 3, 12, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 5, 27, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
    }  # yapf: disable

    def __init__(self,
                 arch='tiny',
                 patch_sizes=[7, 3, 3, 3],
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_values=1e-2,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 norm_eval=False,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 conv_norm_cfg=dict(type='BN', eps=1e-5),
                 attention_types=["GAU", "GAU", "GAU", "GAU",],
                 ffn_types=["Mix", "Mix", "Mix", "Mix",],
                 patchembed_types=["Conv", "Conv", "Conv", "Conv",],
                 with_channel_split=[2, 1, 1],
                 attn_act_gate_cfg=dict(type="GELU"),
                 attn_act_value_cfg=dict(type="GELU"),
                 attn_dw_kernel_size=5,
                 attn_with_dilation=True,
                 attn_with_pointwise=True,
                 attn_with_channel_shuffle=False,
                 attn_decompose_method=None,
                 attn_decompose_position='before',
                 ffn_dwconv_kernel_size=3,
                 ffn_decompose_method='after',
                 ffn_decompose_init_value=0,
                 ffn_decompose_act_cfg=None,
                 ffn_decompose_post_conv=False,
                 block_cfgs=dict(),
                 init_cfg=None):
        super(LAN, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'ffn_ratios'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.ffn_ratios = self.arch_settings['ffn_ratios']
        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.attention_types = attention_types
        self.ffn_types = ffn_types
        assert isinstance(attention_types, (str, list))
        if isinstance(attention_types, str):
            attention_types = [attention_types for i in range(self.num_stages)]
        assert len(attention_types) == self.num_stages
        assert len(ffn_types) == self.num_stages
        assert len(patchembed_types) == self.num_stages

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i > 0 and patchembed_types[i] == "DWConv":
                patch_embed = MiddleEmbedding(
                    in_channels=self.embed_dims[i - 1],
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_cfg=conv_norm_cfg,
                )
            else:
                patch_embed = VANPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    input_size=None,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    padding=(patch_sizes[i] // 2, patch_sizes[i] // 2),
                    norm_cfg=conv_norm_cfg)

            blocks = nn.ModuleList([
                VANBlock(
                    embed_dims=self.embed_dims[i],
                    ffn_ratio=self.ffn_ratios[i],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    norm_cfg=conv_norm_cfg,
                    layer_scale_init_value=init_values,
                    attention_types=self.attention_types[i],
                    ffn_types=self.ffn_types[i],
                    with_channel_split=with_channel_split,
                    attn_act_gate_cfg=attn_act_gate_cfg,
                    attn_act_value_cfg=attn_act_value_cfg,
                    attn_dw_kernel_size=attn_dw_kernel_size,
                    attn_with_dilation=attn_with_dilation,
                    attn_with_pointwise=attn_with_pointwise,
                    attn_with_channel_shuffle=attn_with_channel_shuffle,
                    attn_decompose_method=attn_decompose_method,
                    attn_decompose_position=attn_decompose_position,
                    ffn_dwconv_kernel_size=ffn_dwconv_kernel_size,
                    ffn_decompose_method=ffn_decompose_method,
                    ffn_decompose_init_value=ffn_decompose_init_value,
                    ffn_decompose_act_cfg=ffn_decompose_act_cfg,
                    ffn_decompose_post_conv=ffn_decompose_post_conv,
                    **block_cfgs) for j in range(depth)
            ])
            cur_block_idx += depth
            norm = build_norm_layer(norm_cfg, self.embed_dims[i])[1]

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)

    def init_weights(self, pretrained=None):
        super(LAN, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                for k, m in self.named_modules():
                    if isinstance(m, (nn.Conv2d)):
                        if "offset" in k:  # skip `conv_offset` in DConv
                            if self.init_cfg is not None:
                                m.weight.data.zero_()
                return
            for k, m in self.named_modules():
                if isinstance(m, (nn.Conv2d)):
                    if "offset" in k:  # skip `conv_offset` in DConv
                        if self.init_cfg is not None:
                            m.weight.data.zero_()
                    else:
                        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        fan_out //= m.groups
                        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                        if m.bias is not None:
                            m.bias.data.zero_()
                elif isinstance(m, (nn.Linear)):
                    if not self.is_init:
                        trunc_normal_init(m, mean=0., std=0.02, bias=0)
                elif isinstance(m, (
                    nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = getattr(self, f'patch_embed{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze blocks
            m = getattr(self, f'blocks{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze norm
            m = getattr(self, f'norm{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(-1, *hw_shape,
                          block.out_channels).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)

        return outs

    def train(self, mode=True):
        super(LAN, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                    m.eval()
