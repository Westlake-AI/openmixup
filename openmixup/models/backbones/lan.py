import math
from xml.dom.minidom import Identified
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init

from ..registry import BACKBONES
from .base_backbone import BaseBackbone


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
            padding=1,
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


class GLUFFN(BaseModule):
    """An implementation of FFN with GLU.

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
                 pre_glu=True,
                 glu_balanced_param=False,
                 glu_act_cfg=None,
                 ffn_drop=0.,
                 init_cfg=None):
        super(GLUFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.pre_glu = pre_glu

        if self.pre_glu:  # glu before the FFN activation
            balanced_feedforward_channels = feedforward_channels \
                if not glu_balanced_param else int(feedforward_channels * 4 / 3)
            balanced_feedforward_channels -= (balanced_feedforward_channels % 2)
            self.fc1 = Conv2d(
                in_channels=embed_dims,
                out_channels=balanced_feedforward_channels,
                kernel_size=1)
            self.dwconv = Conv2d(
                in_channels=balanced_feedforward_channels,
                out_channels=balanced_feedforward_channels,
                kernel_size=kernel_size,
                stride=1, padding=1, bias=True,
                groups=balanced_feedforward_channels)
            if glu_act_cfg is not None:
                self.act = build_activation_layer(glu_act_cfg)
                self.glu = None
            else:
                self.act = None
                self.glu = nn.GLU(dim=1)
            self.fc2 = Conv2d(
                in_channels=balanced_feedforward_channels // 2,
                out_channels=embed_dims,
                kernel_size=1)
        else:  # glu after the FFN activation
            balanced_feedforward_channels = feedforward_channels \
                if not glu_balanced_param else int(feedforward_channels * 2 / 3)
            balanced_feedforward_channels -= (balanced_feedforward_channels % 2)
            self.fc1 = Conv2d(
                in_channels=embed_dims,
                out_channels=balanced_feedforward_channels,
                kernel_size=1)
            self.dwconv = Conv2d(
                in_channels=balanced_feedforward_channels,
                out_channels=balanced_feedforward_channels,
                kernel_size=kernel_size,
                stride=1, padding=1, bias=True,
                groups=balanced_feedforward_channels)
            self.act = build_activation_layer(act_cfg)
            self.glu = nn.GLU(dim=1)
            self.fc2 = Conv2d(
                in_channels=balanced_feedforward_channels,
                out_channels=embed_dims * 2,
                kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        if self.pre_glu:
            if self.act is not None:
                f_x, g_x = torch.split(x, self.feedforward_channels // 2, dim=1)
                x = self.act(f_x) * self.act(g_x)  # 4d -> 2d
            else:
                x = self.glu(x)  # 4d -> 2d
        else:
            x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if not self.pre_glu:
            x = self.glu(x)  # 2d -> d
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
                 use_competition=False,
                 act_competition=None,
                 init_cfg=None):
        super(LKA, self).__init__(init_cfg=init_cfg)

        # a spatial local convolution (depth-wise convolution)
        self.DW_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=5,
            padding=2,
            groups=embed_dims)
        # a spatial long-range convolution (depth-wise dilation convolution)
        self.DW_D_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=embed_dims,
            dilation=3)
        # a channel convolution (point-wise convolution)
        self.conv1 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        self.use_competition = use_competition
        if self.use_competition:
            self.act_competition = nn.Softmax(dim=-1) if act_competition == "Softmax" \
                else nn.Sigmoid()
        else:
            self.act_competition = None

    def forward(self, x):
        u = x.clone()
        attn = self.DW_conv(x)
        attn = self.DW_D_conv(attn)
        attn = self.conv1(attn)
        if self.use_competition:
            B, C, H, W = attn.shape
            attn = attn.reshape(B, C, -1)
            attn_sum = 1.0 / (torch.sigmoid(attn).sum(dim=-1, keepdim=True) + 1e-6)
            attn = self.act_competition(attn * attn_sum).reshape(B, C, H, W)

        return u * attn


class SpatialLocalAttention(BaseModule):
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
                 act_cfg=dict(type='GELU'),
                 use_competition=False,
                 act_competition="Softmax",
                 init_cfg=None):
        super(SpatialLocalAttention, self).__init__(init_cfg=init_cfg)

        self.proj_1 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = LKA(embed_dims, use_competition, act_competition)
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


class SpatialGlobalAttention(BaseModule):
    """Global attention module in VANBloack.

    Args:
        embed_dims (int): Number of input channels.

        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 head_dims=None,
                 kernel_size=5,
                 qk_act_cfg=dict(type='Sigmoid'),
                 v_act_cfg=None,
                 v_norm_cfg=None,
                 init_cfg=None):
        super(SpatialGlobalAttention, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.head_dims = head_dims or embed_dims
        self.num_heads = embed_dims // self.head_dims

        self.qkv = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims * 3, kernel_size=1)
        if kernel_size > 0:
            self.DW_conv = Conv2d(
                in_channels=embed_dims * 2,
                out_channels=embed_dims * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=embed_dims * 2)
        else:
            self.DW_conv = nn.Identity()
        self.proj = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.qk_activation = build_activation_layer(qk_act_cfg)
        self.add_one = qk_act_cfg['type'] != "Sigmoid"
        if v_act_cfg is not None:
            self.v_activation = build_activation_layer(v_act_cfg) if \
                v_act_cfg['type'] != "SELU" else nn.SELU()
        else:
            self.v_activation = nn.Identity()
        self.v_norm = build_norm_layer(v_norm_cfg, embed_dims)[1] \
            if v_norm_cfg is not None else nn.Identity()

    def kernel(self, x):
        """ non-neg of the in/output flow """
        x = self.qk_activation(x)
        if self.add_one:
            x = x + 1.
        return x

    def my_sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x):
        shorcut = x.clone()
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)

        qkv = qkv.reshape(B, C, 3, H, W)
        qk = self.DW_conv(qkv[:, :, :2, :, :].reshape(B, C * 2, H, W))
        qk = qk.reshape(B, self.num_heads, self.head_dims,
                        2, N).permute(3, 0, 1, 4, 2).contiguous()
        q, k = qk[0], qk[1]
        v = self.v_activation(self.v_norm(qkv[:, :, 2, :, :]))
        v = v.reshape(B, self.num_heads, self.head_dims,
                      N).permute(0, 1, 3, 2).contiguous()

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
        # competition
        source_competition = torch.softmax(
            conserved_source, dim=-1) * float(k.shape[2])
        # multiply
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * \
            sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        x = (x_update).transpose(1, 2).reshape(B, C, H, W).contiguous()

        x = self.proj(x)
        x = x + shorcut
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
                 local_use_competition=False,
                 local_act_competition="Softmax",
                 global_kernel_size=5,
                 global_qk_act_cfg=dict(type='Sigmoid'),
                 global_v_act_cfg=None,
                 global_v_norm_cfg=None,
                 ffn_glu=False,
                 ffn_pre_glu=False,
                 ffn_glu_balanced_param=False,
                 ffn_glu_act_cfg=None,
                 init_cfg=None):
        super(VANBlock, self).__init__(init_cfg=init_cfg)
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        if attention_types == "LKA":
            self.attn = SpatialLocalAttention(
                embed_dims, act_cfg=act_cfg,
                use_competition=local_use_competition,
                act_competition=local_act_competition)
        elif attention_types == "Flow":
            self.attn = SpatialGlobalAttention(
                embed_dims, head_dims=32, kernel_size=global_kernel_size,
                qk_act_cfg=global_qk_act_cfg,
                v_act_cfg=global_v_act_cfg, v_norm_cfg=global_v_norm_cfg)
        else:
            self.attn = SpatialLocalAttention(embed_dims, act_cfg=act_cfg)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        if not ffn_glu:
            self.mlp = MixFFN(
                embed_dims=embed_dims,
                feedforward_channels=mlp_hidden_dim,
                act_cfg=act_cfg,
                ffn_drop=drop_rate)
        else:
            self.mlp = GLUFFN(
                embed_dims=embed_dims,
                feedforward_channels=mlp_hidden_dim,
                act_cfg=act_cfg,
                pre_glu=ffn_pre_glu,
                glu_balanced_param=ffn_glu_balanced_param,
                glu_act_cfg=ffn_glu_act_cfg,
                ffn_drop=drop_rate)
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


@BACKBONES.register_module()
class LAN(BaseBackbone):
    """Linear Attention Network based on Visual Attention Network.

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
                 attention_types=["LKA", "LKA", "Flow", "Flow",],
                 norm_types=['BN', 'BN', 'LN', 'LN'],
                 local_use_competition=False,
                 local_act_competition="Softmax",
                 global_kernel_size=5,
                 global_qk_act_cfg=dict(type='Sigmoid'),
                 global_v_act_cfg=None,
                 global_v_norm_cfg=None,
                 ffn_glu=False,
                 ffn_pre_glu=False,
                 ffn_glu_balanced_param=False,
                 ffn_glu_act_cfg=None,
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
        assert isinstance(attention_types, (str, list))
        if isinstance(attention_types, str):
            attention_types = [attention_types for i in range(self.num_stages)]
        assert len(attention_types) == self.num_stages

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
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
                    norm_cfg=conv_norm_cfg if "BN" in norm_types[i] else dict(type="LN2d"),
                    layer_scale_init_value=init_values,
                    attention_types=self.attention_types[i],
                    local_use_competition=local_use_competition,
                    local_act_competition=local_act_competition,
                    global_kernel_size=global_kernel_size,
                    global_qk_act_cfg=global_qk_act_cfg,
                    global_v_act_cfg=global_v_act_cfg,
                    global_v_norm_cfg=global_v_norm_cfg,
                    ffn_glu=ffn_glu,
                    ffn_pre_glu=ffn_pre_glu,
                    ffn_glu_balanced_param=ffn_glu_balanced_param,
                    ffn_glu_act_cfg=ffn_glu_act_cfg,
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
                return
            for m in self.modules():
                if isinstance(m, (nn.Conv2d)):
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
