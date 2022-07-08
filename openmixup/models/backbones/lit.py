import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init
from mmcv.runner.base_module import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..utils import DeformablePatchMerging, HiLoAttention
from ..registry import BACKBONES
from .base_backbone import BaseBackbone
from .vision_transformer import TransformerEncoderLayer


class DWConv(nn.Module):
    """An implementation of depth-wise conv for FFN

    Args:
        embed_dims (int): The feature dimension.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
    """

    def __init__(self, embed_dims=768, kernel_size=3):
        super(DWConv, self).__init__()
        self.dwconv = Conv2d(embed_dims, embed_dims, kernel_size,
            stride=1, padding=kernel_size // 2, bias=True, groups=embed_dims)

    def forward(self, x):
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(1, 2).view(B, C, H, W).contiguous()
            x = self.dwconv(x)
            x = x.flatten(2).transpose(1, 2).contiguous()
        elif x.dim() == 4:
            x = self.dwconv(x)
        return x


class DWConvMLP(BaseModule):
    """An implementation of Depth-wise conv FFN

    Args:
        in_features (int): The feature dimension.
        hidden_features (int): The hidden dimension of FFNs.
        out_features (int): The output dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 kernel_size=3,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.0,
                 init_cfg=None):
        super(DWConvMLP, self).__init__(init_cfg=init_cfg)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        kernel_size = kernel_size or 3
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, kernel_size)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvFFNBlock(BaseModule):
    """Implement of Conv-based block in LIT (FFN only).

    Args:
        embed_dims (int): The feature dimension.
        mlp_ratio (int): The hidden dimension for FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
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
                 mlp_ratio=4.,
                 kernel_size=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_values=0.0,
                 init_cfg=None,
                 **kwargs):
        super(ConvFFNBlock, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # feed forward MLP
        self.ffn = DWConvMLP(
            in_features=self.embed_dims,
            hidden_features=int(self.embed_dims * mlp_ratio),
            kernel_size=kernel_size,
            ffn_drop=drop_rate,
            act_cfg=act_cfg
        )
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if init_values > 0:
            self.gamma = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
        else:
            self.gamma = None

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        if self.gamma is not None:
            x = x + self.drop_path(self.gamma * self.ffn(self.norm1(x)))
        else:
            x = x + self.drop_path(self.ffn(self.norm1(x)))
        return x


class HiLoBlock(BaseModule):
    """Implement of HiLo-attnetion-based Block in LIT.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratio (int): The hidden dimension for FFNs.
        window_size (int | None): Local window size of attention.
        alpha (float): Ratio to split the attention to high and low parts.
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
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 window_size=2,
                 alpha=0.5,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_values=0.0,
                 with_cp=False,
                 init_cfg=None,
                 **kwargs):
        super(HiLoBlock, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.with_cp = with_cp

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # self-attention
        assert 1 <= window_size and 0 <= alpha <= 1
        self.attn = HiLoAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            alpha=alpha,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop_rate, proj_drop=drop_rate)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        # feed forward MLP
        self.ffn = DWConvMLP(
            in_features=embed_dims,
            hidden_features=int(embed_dims * mlp_ratio),
            kernel_size=3,
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

        def _inner_forward(x):
            if self.gamma_1 is not None:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@BACKBONES.register_module()
class LIT(BaseBackbone):
    """LITv1 and LITv2 Transformer.

    PyTorch implements of LITv1 : `Less is More: Pay Less Attention in Vision
        Transformers <https://arxiv.org/abs/2105.14217>`_
    and LITv2 : `Fast Vision Transformers with HiLo Attention
        <https://arxiv.org/abs/2205.13213>`_

    Modified from the `official repo
    <https://github.com/ziplab/LIT/tree/main/classification> and
    <https://github.com/ziplab/LITv2/tree/main/classification>`_

    Args:
        arch (str | dict): LIT architecture.
            If use string, choose from 'tiny', 'small', and 'base'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (int): The number of head in each stage.
            - **mlp_ratio** (bool): The ratio of mlp hidden dim to embedding dim.

            We provide UniFormer-Tiny (based on VAN-Tiny) in addition to the
            original paper. Defaults to 'small'.
        input_size (int | tuple): The expected input image or sequence shape.
            We don't support dynamic input shape, please set the argument to the
            true input shape. Defaults to 224.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to 3, means the last stage.
        window_size (int | None): Local window size of attention.
        init_value (float): Init value for Layer Scale. Defaults to 1e-6.
        alpha (float): Ratio to split the attention to high and low parts.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for self-attention normalization layer.
            Defaults to ``dict(type='LN')``.
        act_cfg (dict): The config dict for activation after each convolution.
            Defaults to ``dict(type='GELU')``.
        conv_norm_cfg (dict): Config dict for convolution normalization layer.
            Defaults to ``dict(type='BN')``.
        attention_types (str | list): Type of spatial attention in each stages.
            For example, LITv2 uses [None, None, "HiLo", "HiLo"] by default.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 4, 6, 3],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratio': [8, 8, 4, 4],
                        }),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [96, 192, 384, 768],
                         'depths': [2, 2, 6, 2],
                         'num_heads': [3, 6, 12, 24],
                         'mlp_ratio': [4, 4,  4,  4],
                        }),
        **dict.fromkeys(['m', 'medium'],
                        {'embed_dims': [96, 192, 384, 768],
                         'depths': [2, 2, 18, 2],
                         'num_heads': [3, 6, 12, 24],
                         'mlp_ratio': [4, 4,  4,  4],
                        }),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [128, 256, 512, 1024],
                         'depths': [2, 2, 18, 2],
                         'num_heads': [4, 8, 16, 32],
                         'mlp_ratio': [4, 4,  4,  4],
                        }),
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 in_channels=3,
                 stem_patch_size=4,
                 out_indices=(3,),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 window_size=[0, 0, 1, 1,],
                 alpha=0.5,
                 init_values=0.0,
                 norm_cfg=dict(type='LN', eps=1e-5),
                 act_cfg=dict(type='GELU'),
                 conv_norm_cfg=dict(type='BN', eps=1e-5),
                 attention_types=[None, None, "MHSA", "MHSA",],
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None,
                 **kwargs):
        super(LIT, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'num_heads', 'mlp_ratio'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
            self.arch = 'small'

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratio = self.arch_settings['mlp_ratio']
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
        self.alpha = alpha
        self.window_size = window_size
        assert isinstance(window_size, (int, list))
        if isinstance(window_size, int):
            self.window_size = [max(1, window_size) for i in range(self.num_stages)]
        assert len(window_size) == self.num_stages

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            # build patch embedding
            if i == 0:
                patch_embed = PatchEmbed(
                    in_channels=in_channels,
                    input_size=None,
                    embed_dims=self.embed_dims[i],
                    kernel_size=stem_patch_size,
                    stride=stem_patch_size,
                    padding=0,
                    norm_cfg=norm_cfg,
                )
            else:
                patch_embed = DeformablePatchMerging(
                    in_channels=self.embed_dims[i - 1],
                    out_channels=self.embed_dims[i],
                    kernel_size=2, stride=2,
                    padding=0,
                    norm_cfg=conv_norm_cfg,
                    act_cfg=act_cfg,
                )

            # build spatial mixing block
            if self.attention_types[i] is None:
                blocks = nn.ModuleList([
                    ConvFFNBlock(
                        embed_dims=self.embed_dims[i],
                        mlp_ratio=self.mlp_ratio[i],
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur_block_idx + j],
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        init_values=init_values,
                    ) for j in range(depth)
                ])
            elif self.attention_types[i] == "MHSA":
                blocks = nn.ModuleList([
                    TransformerEncoderLayer(
                        embed_dims=self.embed_dims[i],
                        num_heads=self.num_heads[i],
                        feedforward_channels=self.mlp_ratio[i],
                        window_size=None,
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur_block_idx + j],
                        qkv_bias=qkv_bias,
                        norm_cfg=norm_cfg,
                        init_values=init_values,
                    ) for j in range(depth)
                ])
            elif self.attention_types[i] == "HiLo":
                blocks = nn.ModuleList([
                    HiLoBlock(
                        embed_dims=self.embed_dims[i],
                        num_heads=self.num_heads[i],
                        mlp_ratio=self.mlp_ratio[i],
                        window_size=self.window_size[i],
                        alpha=alpha,
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur_block_idx + j],
                        qkv_bias=qkv_bias,
                        norm_cfg=norm_cfg,
                        init_values=init_values,
                    ) for j in range(depth)
                ])
            else:
                raise NotImplementedError

            cur_block_idx += depth
            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)

        for i in self.out_indices:
            if i < 0:
                continue
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg, self.embed_dims[i])[1]
            else:
                norm_layer = nn.Identity()
            self.add_module(f'norm{i}', norm_layer)

    def init_weights(self, pretrained=None):
        super(LIT, self).init_weights(pretrained)

        if pretrained is None:
            for k, m in self.named_modules():
                if isinstance(m, (nn.Conv2d)):
                    if "offset" in k:  # skip `conv_offset` in DConv
                        if self.init_cfg is not None:
                            m.weight.data.zero_()
                    elif self.init_cfg is not None:
                        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        fan_out //= m.groups
                        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                        if m.bias is not None:
                            m.bias.data.zero_()
                if self.init_cfg is None:
                    if isinstance(m, (nn.Linear)):
                        trunc_normal_init(m, mean=0., std=0.02, bias=0)
                    elif isinstance(m, (
                        nn.LayerNorm, _BatchNorm, nn.GroupNorm, nn.SyncBatchNorm)):
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
            if i in self.out_indices and i > 0:
                m = getattr(self, f'norm{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')

            if x.dim() == 3:
                x = x.view(x.shape[0],
                           *hw_shape, -1).permute(0, 3, 1, 2).contiguous()

            x, hw_shape = patch_embed(x)
            if i == 0:
                x = self.drop_after_pos(x)
            for block in blocks:
                x = block(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x = norm_layer(x)
                x = x.view(x.shape[0],
                           *hw_shape, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(x)

        return outs

    def train(self, mode=True):
        super(LIT, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
