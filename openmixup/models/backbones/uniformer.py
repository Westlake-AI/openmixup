import math
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init

from ..utils import MultiheadAttention, MultiheadAttentionWithRPE
from ..registry import BACKBONES
from .base_backbone import BaseBackbone
from .van import VANBlock


class MLP(nn.Module):
    """An implementation of vanilla FFN

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
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMLP(nn.Module):
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
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d(in_features, hidden_features, 1)
        self.fc2 = Conv2d(hidden_features, out_features, 1)
        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBlock(nn.Module):
    """Implement of Conv-based block in Uniformer.

    Args:
        embed_dims (int): The feature dimension.
        mlp_ratio (int): The hidden dimension for FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            spatial attention. Defaults to 5.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        init_values (float): The init values of gamma. Defaults to 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 mlp_ratio=4.,
                 kernel_size=5,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'),
                 init_values=1e-6,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # positional encoding
        self.pos_embed = Conv2d(
            embed_dims, embed_dims, 3, padding=1, groups=embed_dims)

        # spatial attention
        self.conv1 = Conv2d(embed_dims, embed_dims, 1)
        self.conv2 = Conv2d(embed_dims, embed_dims, 1)
        self.attn = Conv2d(embed_dims, embed_dims, kernel_size,
            padding=kernel_size // 2, groups=embed_dims)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        # feed forward MLP
        self.ffn = ConvMLP(
            in_features=self.embed_dims,
            hidden_features=int(self.embed_dims * mlp_ratio),
            ffn_drop=drop_rate,
            act_cfg=act_cfg
        )

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        x = x + self.pos_embed(x)
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.conv2(self.attn(self.conv1(self.norm1(x)))))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class SABlock(nn.Module):
    """Implement of Self-attnetion-based Block in Uniformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratio (int): The hidden dimension for FFNs.
        window_size (int | None): Local window size of attention.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        qk_scale (int | None): Scale of the qk attention. Defaults to None.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_values (float): The init values of gamma. Defaults to 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 window_size=None,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_values=1e-6,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # positional encoding
        self.pos_embed = Conv2d(
            embed_dims, embed_dims, 3, padding=1, groups=embed_dims)

        # self-attention
        if window_size is None:
            # attention without relative position bias
            self.attn = MultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, proj_drop=drop_rate)
        else:
            # attention with relative position bias
            self.attn = MultiheadAttentionWithRPE(
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, proj_drop=drop_rate)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        # feed forward MLP
        self.ffn = MLP(
            in_features=embed_dims,
            hidden_features=int(embed_dims * mlp_ratio),
            ffn_drop=drop_rate,
            act_cfg=act_cfg)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W).contiguous()
        return x


class ConvEmbedding(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        out_features (int): The output dimension of FFNs.
        kernel_size (int): The conv kernel size of middle patch embedding.
            Defaults to 3.
        stride_size (int): The conv stride of middle patch embedding.
            Defaults to 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride_size=2,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'),
                ):
        super(ConvEmbedding, self).__init__()
        
        self.projection = nn.Sequential(
            Conv2d(in_channels, out_channels // 2, kernel_size=kernel_size,
                stride=stride_size, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            Conv2d(out_channels // 2, out_channels, kernel_size=kernel_size,
                stride=stride_size, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class MiddleEmbedding(nn.Module):
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
                 out_channels,
                 kernel_size=3,
                 stride_size=2,
                 norm_cfg=dict(type='BN'),
                ):
        super(MiddleEmbedding, self).__init__()
        
        self.projection = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride_size, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.projection(x)
        return x


@BACKBONES.register_module()
class UniFormer(BaseBackbone):
    """Unified Transformer.

    A PyTorch implement of : `UniFormer: Unifying Convolution and Self-attention
    for Visual Recognition <https://arxiv.org/abs/2201.04676>`_

    Modified from the `official repo
    <https://github.com/Sense-X/UniFormer/tree/main/image_classification>`_

    Args:
        arch (str | dict): UniFormer architecture.
            If use string, choose from 'small' and 'base'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **head_dim** (int): The dimensions of each head.
            - **patch_strides** (List[int]): The stride of each stage.
            - **conv_stem** (bool): Whether to use conv-stem.

            We provide UniFormer-Tiny (based on VAN-Tiny) in addition to the
            original paper. Defaults to 'small'.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to 3, means the last stage.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        init_value (float): Init value for Layer Scale. Defaults to 1e-6.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        conv_stem (bool): whether use overlapped patch stem.
        conv_kernel_size (int | list): The conv kernel size in the PatchEmbed.
            Defaults to 3, which is used when conv_stem=True.
        attn_kernel_size (int): The conv kernel size in the ConvBlock as the
            spatial attention. Defaults to 5.
        norm_cfg (dict): Config dict for self-attention normalization layer.
            Defaults to ``dict(type='LN')``.
        act_cfg (dict): The config dict for activation after each convolution.
            Defaults to ``dict(type='GELU')``.
        conv_norm_cfg (dict): Config dict for convolution normalization layer.
            Defaults to ``dict(type='BN')``.
        attention_types (str | list): Type of spatial attention in each stages.
            UniFormer uses ["Conv", "Conv", "MHSA", "MHSA"] by default.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 160, 256],
                         'depths': [3, 4, 8, 3],
                         'head_dim': 32,
                         'patch_strides': [4, 2, 2, 2],
                         'conv_stem': False,
                        }),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 4, 8, 3],
                         'head_dim': 64,
                         'patch_strides': [4, 2, 2, 2],
                         'conv_stem': False,
                        }),
        **dict.fromkeys(['s+', 'small_plus'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 5, 9, 3],
                         'head_dim': 32,
                         'patch_strides': [2, 2, 2, 2],
                         'conv_stem': True,
                        }),
        **dict.fromkeys(['s+_dim64', 'small_plus_dim64'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 5, 9, 3],
                         'head_dim': 64,
                         'patch_strides': [2, 2, 2, 2],
                         'conv_stem': True,
                        }),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [5, 8, 20, 7],
                         'head_dim': 64,
                         'patch_strides': [4, 2, 2, 2],
                         'conv_stem': False,
                        }),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [128, 192, 448, 640],
                         'depths': [5, 10, 24, 7],
                         'head_dim': 64,
                         'patch_strides': [4, 2, 2, 2],
                         'conv_stem': False,
                        }),
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 in_channels=3,
                 out_indices=(3,),
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 init_values=1e-6,
                 conv_kernel_size=3,
                 attn_kernel_size=5,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 conv_norm_cfg=dict(type='BN'),
                 attention_types=["Conv", "Conv", "MHSA", "MHSA",],
                 final_norm=True,
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None,
                 **kwargs):
        super(UniFormer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'head_dim', 'patch_strides', 'conv_stem'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
            self.arch = 'small'

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.head_dim = self.arch_settings['head_dim']
        self.patch_strides = self.arch_settings['patch_strides']
        self.conv_stem = self.arch_settings['conv_stem']
        self.mlp_ratio = mlp_ratio
        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        assert isinstance(out_indices, (int, tuple, list))
        if isinstance(out_indices, int):
            self.out_indices = [out_indices]

        self.attention_types = attention_types
        assert isinstance(attention_types, (str, list))
        if isinstance(attention_types, str):
            attention_types = [attention_types for i in range(self.num_stages)]
        assert len(attention_types) == self.num_stages
        assert isinstance(conv_kernel_size, (int, tuple, list))
        if isinstance(conv_kernel_size, int):
            conv_kernel_size = [conv_kernel_size for i in range(self.num_stages)]
        assert len(conv_kernel_size) == self.num_stages

        if "BN" in norm_cfg["type"]:
            norm_cfg["type"] = "BN1d"
        if "LN" in conv_norm_cfg["type"]:
            conv_norm_cfg["type"] = "LN2d"

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        num_heads = [dim // self.head_dim for dim in self.embed_dims]

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            # build patch embedding
            if self.conv_stem:
                if i == 0:
                    patch_embed = ConvEmbedding(
                        in_channels=in_channels, out_channels=self.embed_dims[i],
                        kernel_size=conv_kernel_size[i], stride_size=self.patch_strides[i],
                        norm_cfg=conv_norm_cfg, act_cfg=act_cfg,
                    )
                else:
                    patch_embed = MiddleEmbedding(
                        in_channels=self.embed_dims[i - 1], out_channels=self.embed_dims[i],
                        kernel_size=conv_kernel_size[i], stride_size=self.patch_strides[i],
                        norm_cfg=conv_norm_cfg,
                    )
            else:
                patch_embed = PatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    input_size=None,
                    embed_dims=self.embed_dims[i],
                    kernel_size=self.patch_strides[i], stride=self.patch_strides[i],
                    padding=0 if self.patch_strides[i] % 2 == 0 else 'corner',
                    norm_cfg=norm_cfg,
                )

            # build spatial mixing block
            if self.attention_types[i] == "Conv":
                blocks = nn.ModuleList([
                    ConvBlock(
                        embed_dims=self.embed_dims[i],
                        mlp_ratio=mlp_ratio,
                        kernel_size=attn_kernel_size,
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur_block_idx + j],
                        norm_cfg=conv_norm_cfg,
                        init_values=init_values,
                    ) for j in range(depth)
                ])
            elif self.attention_types[i] == "MHSA":
                blocks = nn.ModuleList([
                    SABlock(
                        embed_dims=self.embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratio,
                        window_size=None,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[cur_block_idx + j],
                        norm_cfg=norm_cfg,
                        init_values=init_values,
                    ) for j in range(depth)
                ])
            elif self.attention_types[i] == "VAN":
                blocks = nn.ModuleList([
                    VANBlock(
                        embed_dims=self.embed_dims[i],
                        ffn_ratio=mlp_ratio,
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur_block_idx + j],
                        norm_cfg=conv_norm_cfg,
                        layer_scale_init_value=init_values,
                    ) for j in range(depth)
                ])
            else:
                raise NotImplementedError

            cur_block_idx += depth
            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)

        self.final_norm = final_norm
        if self.final_norm:
            for i in self.out_indices:
                if i < 0:
                    continue
                norm_layer = build_norm_layer(conv_norm_cfg, self.embed_dims[i])[1]
                self.add_module(f'norm{i}', norm_layer)

    def init_weights(self, pretrained=None):
        super(UniFormer, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0)
                elif isinstance(m, (
                    nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()

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
            if i in self.out_indices and i > 0:
                if self.final_norm:
                    m = getattr(self, f'norm{i}')
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            
            x = patch_embed(x)
            if len(x) == 2:
                x, hw_shape = x  # patch_embed
                x = x.reshape(x.shape[0],
                              *hw_shape, -1).permute(0, 3, 1, 2).contiguous()
            if i == 0:
                x = self.drop_after_pos(x)
            for block in blocks:
                x = block(x)
            if i in self.out_indices:
                if self.final_norm:
                    norm_layer = getattr(self, f'norm{i}')
                    x = norm_layer(x)
                outs.append(x)

        return outs

    def train(self, mode=True):
        super(UniFormer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
