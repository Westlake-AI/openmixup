import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init
from mmcv.runner.base_module import BaseModule

from ..registry import BACKBONES
from .base_backbone import BaseBackbone


class MLP(BaseModule):
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
                 ffn_drop=0.0,
                 init_cfg=None):
        super(MLP, self).__init__(init_cfg=init_cfg)

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


class SRAttention(BaseModule):
    """Spatial-reduction Attention Module (SRA) in PVT.

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
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        sr_ratio (float): Spatial reduction ratio. Defaults to 1.
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
                 qk_scale=None,
                 proj_bias=True,
                 sr_ratio=1,
                 init_cfg=None):
        super(SRAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.q = nn.Linear(self.input_dims, self.embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(self.input_dims, self.embed_dims * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PVTBlock(BaseModule):
    """Implements of PVT module.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        mlp_ratio (int): The number of fully-connected layers for FFNs.
            Defaults to 4.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        sr_ratio (float): Spatial reduction ratio. Defaults to 1.
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
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 mlp_ratio=4,
                 qkv_bias=True,
                 sr_ratio=1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_values=0,
                 init_cfg=None):
        super(PVTBlock, self).__init__(init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = SRAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias,
            sr_ratio=sr_ratio)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = MLP(
            in_features=embed_dims,
            hidden_features=embed_dims * mlp_ratio,
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

    def forward(self, x, H, W):
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


@BACKBONES.register_module()
class PyramidVisionTransformer(BaseBackbone):
    """Pyramid Vision Transformer (PVT).

    A PyTorch implement of : `Pyramid Vision Transformer: A Versatile Backbone
    for Dense Prediction without Convolutions <https://arxiv.org/abs/2102.12122>`_

    Modified from the `official repo <https://github.com/whai362/PVT>`_

    Args:
        arch (str | dict): UniFormer architecture.
            If use string, choose from 'small' and 'base'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (List[int]): The number of head in each stage.
            - **mlp_ratio** (List[int]): The MLP ratio in each stage.
            - **sr_ratios** (List[int]): The spatial reduction ration of each stage.

            Defaults to 'small'.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to 3, means the last stage.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        init_value (float): Init value for Layer Scale. Defaults to 0.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for self-attention normalization layer.
            Defaults to ``dict(type='LN')``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        act_cfg (dict): The config dict for activation after each convolution.
            Defaults to ``dict(type='GELU')``.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 2, 2, 2],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratio': [8, 8, 4, 4],
                         'sr_ratios': [8, 4, 2, 1],
                        }),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 4, 6, 3],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratio': [8, 8, 4, 4],
                         'sr_ratios': [8, 4, 2, 1],
                        }),
        **dict.fromkeys(['m', 'medium'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 4, 18, 3],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratio': [8, 8, 4, 4],
                         'sr_ratios': [8, 4, 2, 1],
                        }),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 8, 27, 3],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratio': [8, 8, 4, 4],
                         'sr_ratios': [8, 4, 2, 1],
                        }),
        **dict.fromkeys(['h', 'huge_v2'],
                        {'embed_dims': [128, 256, 512, 768],
                         'depths': [3, 10, 60, 3],
                         'num_heads': [2, 4, 8, 12],
                         'mlp_ratio': [8, 8, 4, 4],
                         'sr_ratios': [8, 4, 2, 1],
                        }),
    }
    # Some structures have multiple extra tokens, like DeiT.
    num_extra_tokens = 1  # cls_token

    def __init__(self,
                 arch='small',
                 in_channels=3,
                 img_size=224,
                 patch_size=4,
                 out_indices=(3,),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 init_values=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 with_cls_token=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs):
        super(PyramidVisionTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'num_heads', 'mlp_ratio', 'sr_ratios'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
            self.arch = 'small'

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratio = self.arch_settings['mlp_ratio']
        self.sr_ratios = self.arch_settings['sr_ratios']

        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.interpolate_mode = interpolate_mode
        assert isinstance(out_indices, (int, tuple, list))
        if isinstance(out_indices, int):
            self.out_indices = [out_indices]

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            # build patch embedding
            _patch_cfg = dict(
                in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                input_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                embed_dims=self.embed_dims[i],
                conv_type='Conv2d',
                kernel_size=patch_size if i == 0 else 2,
                stride=patch_size if i == 0 else 2,
                padding=0, norm_cfg=norm_cfg,
            )
            patch_embed = PatchEmbed(**_patch_cfg)
            num_patches = patch_embed.init_out_size[0] * patch_embed.init_out_size[1]
            num_patches = num_patches if i != self.num_stages - 1 else num_patches + 1
            pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            # build spatial mixing block
            blocks = nn.ModuleList([
                PVTBlock(
                    embed_dims=self.embed_dims[i],
                    num_heads=self.num_heads[i],
                    mlp_ratio=self.mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    sr_ratio=self.sr_ratios[i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    init_values=init_values,
                ) for j in range(depth)
            ])

            cur_block_idx += depth
            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'pos_drop{i + 1}', pos_drop)
            self.add_module(f'blocks{i + 1}', blocks)
            setattr(self, f"pos_embed{i + 1}", pos_embed)

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims[-1]))

        # Set output norm
        for i in self.out_indices:
            norm_layer = build_norm_layer(norm_cfg, self.embed_dims[i])[1]
            self.add_module(f'norm{i}', norm_layer)

    def init_weights(self, pretrained=None):
        super(PyramidVisionTransformer, self).init_weights(pretrained)

        if pretrained is None:
            nn.init.trunc_normal_(self.cls_token, mean=0, std=.02)
            for i in range(self.num_stages):
                pos_embed = getattr(self, f"pos_embed{i + 1}")
                nn.init.trunc_normal_(pos_embed, mean=0, std=.02)
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                    constant_init(m, val=1, bias=0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if (H, W) == patch_embed.init_out_size:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, *patch_embed.init_out_size, -1).permute(0, 3, 1, 2),
                size=(H, W),
                mode=self.interpolate_mode).reshape(1, -1, H * W).permute(0, 2, 1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.cls_token.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = getattr(self, f'patch_embed{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze pos embed
            m = getattr(self, f"pos_embed{i + 1}")
            m.eval()
            m.requires_grad = False

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
        B = x.size(0)
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            blocks = getattr(self, f"blocks{i + 1}")

            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for block in blocks:
                x = block(x, H, W)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                _x = norm_layer(x)

                if self.with_cls_token and i == self.num_stages - 1:
                    patch_token = _x[:, 1:].reshape(B, H, W, -1)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = _x[:, 0]
                else:
                    patch_token = _x.reshape(B, H, W, -1)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.output_cls_token and i == self.num_stages - 1:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)

            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return outs

    def train(self, mode=True):
        super(PyramidVisionTransformer, self).train(mode)
        self._freeze_stages()
