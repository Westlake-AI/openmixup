import math

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init
from mmcv.runner.base_module import BaseModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat, reduce
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(BaseModule):
    def __init__(self,
                in_features,
                hidden_features=None,
                out_features=None,
                act_cfg=dict(type='GELU'),
                drop=0.,
                init_cfg=None):
        super(Mlp, self).__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BasicLayer(BaseModule):
    def __init__(self,
                dim,
                input_resolution,
                depth,
                num_heads,
                patch_size=4,
                merge_ratio=2,
                mlp_ratio=4., 
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=None,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                downsample=True,
                use_checkpoint=False,
                init_cfg=None):
        super(BasicLayer, self).__init__(init_cfg=None)

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        drop_path = 0. if drop_path is None else drop_path

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, input_resolution=input_resolution,
                             num_heads=num_heads, patch_size=patch_size,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop, attn_drop=attn_drop,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             act_cfg=act_cfg, norm_cfg=norm_cfg)
            for i in range(depth)])

        merge_size = int(patch_size * merge_ratio)
        self.road_merge = RoadMerge(patch_size=merge_size, in_chans=dim, downsample=downsample)

    def forward(self, x, h, w, return_x=False):

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, h, w)
            else:
                x = blk(x, h, w)

        x, h, w = self.road_merge(x, h, w)
        if return_x:
            return x
        return x, h, w


class Attention(BaseModule):
    def __init__(self,
                dim,
                feature_size,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.,
                proj_drop=0.,
                init_cfg=None):
        super(Attention, self).__init__(init_cfg=init_cfg)
        self.dim = dim
        self.feature_size = feature_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * feature_size[0] - 1) * (2 * feature_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.feature_size[0])
        coords_w = torch.arange(self.feature_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.feature_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.feature_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.feature_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.feature_size[0] * self.feature_size[1], self.feature_size[0] * self.feature_size[1],
            -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvBNGelu(nn.Module):
    def __init__(self,
                in_channel,
                out_channel,
                kernel_size,
                stride_size,
                padding=1):
        """
        build the conv3x3 + gelu + bn module
        """
        super(ConvBNGelu, self).__init__()
        self.kernel_size = to_2tuple(kernel_size)
        self.stride_size = to_2tuple(stride_size)
        self.padding_size = to_2tuple(padding)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv_bn_gelu = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=self.kernel_size,
                      stride=self.stride_size,
                      padding=self.padding_size),
            nn.BatchNorm2d(self.out_channel),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_bn_gelu(x)
        return x


class Conv3dBNGelu(nn.Module):
    def __init__(self,
                in_channel,
                out_channel,
                kernel_size,
                stride_size,
                padding,
                act=True):
        super(Conv3dBNGelu, self).__init__()

        self.act = act
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride_size,
                      padding=padding),
            nn.BatchNorm3d(out_channel))
        if self.act:
            self.action = nn.GELU()

    def forward(self, x):

        x = self.conv3(x)
        if self.act:
            x = self.action(x)

        return x


class Conv3Res(nn.Module):
    def __init__(self,
                in_channel,
                layers_num=2):
        super(Conv3Res, self).__init__()
        self.layers_num = layers_num

        self.conv3_res_list = nn.ModuleList([nn.Sequential(
            Conv3dBNGelu(in_channel, in_channel // 4, kernel_size=(1, 1, 1), stride_size=(1, 1, 1),
                         padding=(0, 0, 0)),
            Conv3dBNGelu(in_channel // 4, in_channel // 4, kernel_size=(1, 3, 3), stride_size=(1, 1, 1),
                         padding=(0, 1, 1)),
            Conv3dBNGelu(in_channel // 4, in_channel, kernel_size=(1, 1, 1), stride_size=(1, 1, 1),
                         padding=(0, 0, 0), act=False))
            for _ in range(self.layers_num)]
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        for i in range(self.layers_num):
            x = self.gelu(x + self.conv3_res_list[i](x))
        return x


class ConvDW3x3(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ConvDW3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=to_2tuple(kernel_size),
            padding=to_2tuple(1),
            groups=dim)

    def forward(self, x):
        x = self.conv(x)
        return x


class RMTransformer(BaseModule):
    """Attention Label Learning to Enhance Interactive Vein Transformer for Palm-Vein Recognition.

    A PyTorch implement of : `Attention Label Learning to Enhance Interactive Vein Transformer 
    for Palm-Vein Recognition <https://ieeexplore.ieee.org/document/10479213>`_
    
    Published on IEEE Transactions on Biometrics, Behavior, and Identity Science 2024

    Modified by Xin Jin`_

    """
    def __init__(self, 
                img_size=(224, 224), 
                patch_size=2, 
                in_chans=3,
                num_classes=1000,
                embed_dim=48,
                depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 24),
                merge_ratio=2, 
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                # norm_layer=nn.LayerNorm,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                ape=False,
                patch_norm=True,
                use_checkpoint=False, 
                init_cfg=None,
                **kwargs):
        super(RMTransformer, self).__init__(init_cfg)

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.merge_ratio = merge_ratio
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.rmt_stem = RMTStem(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, layers_num=2)

        feature_h, feature_w = img_size[0] // 4, img_size[1] // 4
        feature_size = feature_h * feature_w

        patches_resolution = (feature_h // patch_size, feature_w // patch_size)
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, feature_size, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** layer_idx),
                               input_resolution=(math.ceil(patches_resolution[0] / (2 ** layer_idx)),
                                                 math.ceil(patches_resolution[1] / (2 ** layer_idx))),
                               depth=depths[layer_idx],
                               num_heads=num_heads[layer_idx],
                               patch_size=patch_size,
                               merge_ratio=merge_ratio,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:layer_idx]):sum(depths[:layer_idx + 1])],
                               act_cfg=act_cfg, norm_cfg=norm_cfg,
                               downsample=True if (layer_idx < self.num_layers - 1) else False,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i < 0:
                continue
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def forward_features(self, x):
        x = self.rmt_stem(x)

        if self.ape:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = x + self.absolute_pos_embed
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.pos_drop(x)

        (h, w) = self.patches_resolution
        length = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i == length - 1:
                x = layer(x, h, w, return_x=True)
            else:
                x, h, w = layer(x, h, w)

        return x

    def forward(self, x, return_feature=False):
        out = self.forward_features(x)
        if return_feature:
            return out
        return out

    def train(self, mode=True):
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                    m.eval()


class RMTStem(nn.Module):
    """
    make the model conv stem module
    """
    def __init__(self,
                in_channel=3,
                out_channel=48,
                kernel_size=3,
                layers_num=2):
        super(RMTStem, self).__init__()
        self.layers_num = layers_num

        self.conv_bn_gelu_downsample = ConvBNGelu(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            stride_size=to_2tuple(2)
        )
        self.conv_bn_gelu_list = nn.ModuleList(
            [ConvBNGelu(in_channel=out_channel, out_channel=out_channel, kernel_size=kernel_size, stride_size=1) for _
             in range(self.layers_num)]
        )

        self.patch_downsample = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=to_2tuple(3),
                                          stride=to_2tuple(2), padding=1)

    def forward(self, x):
        x = self.conv_bn_gelu_downsample(x)
        for i in range(self.layers_num):
            x = self.conv_bn_gelu_list[i](x)
        x = self.patch_downsample(x)
        return x


class RoadMerge(nn.Module):
    r"""
    """

    def __init__(self, patch_size=8, in_chans=48, layers_num=2, downsample=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.layers_num = layers_num
        self.down = downsample

        # self.conv3_res = Conv3Res(in_channel=in_chans, layers_num=layers_num)
        if self.down:
            self.downsample = Conv3dBNGelu(in_channel=in_chans, out_channel=2 * in_chans, kernel_size=(1, 2, 2),
                                           stride_size=(1, 2, 2), padding=(0, 0, 0), act=False)

    def forward(self, x, h, w):
        """
            x: B, H*W, C
        """
        B, C, H, W = x.shape
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))
            B, C, H, W = x.shape

        h, w = H // self.patch_size[0], W // self.patch_size[1]

        x = rearrange(x, 'b c (h p1) (w p2) -> b c (h w) p1 p2', p1=self.patch_size[0], p2=self.patch_size[1])
        # x = self.conv3_res(x)
        if self.down:
            x = self.downsample(x)

        x = rearrange(x, 'b c (h w) p1 p2 -> b c (h p1) (w p2)', h=h, w=w)

        return x, h, w


class TransformerBlock(BaseModule):
    def __init__(self,
                dim,
                input_resolution,
                num_heads,
                patch_size=4,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                init_cfg=None):
        super(TransformerBlock, self).__init__(init_cfg)

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio

        cn_dim = dim // 4
        self.conv1x1_bn_gelu = ConvBNGelu(
            in_channel=dim,
            out_channel=cn_dim,
            kernel_size=1,
            stride_size=1,
            padding=0
        )
        self.conv3x3_dw = ConvDW3x3(dim=cn_dim)
        self.act = nn.Sequential(
            nn.BatchNorm2d(cn_dim),
            nn.GELU()
        )
        at_dim = cn_dim * self.patch_size[0] * self.patch_size[1]
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.dim, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = Attention(at_dim, feature_size=self.input_resolution, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.dim, postfix=1)
        self.add_module(self.norm2_name, norm2)

        mlp_hidden_dim = int(at_dim * mlp_ratio)
        self.mlp = Mlp(in_features=at_dim, hidden_features=mlp_hidden_dim, act_cfg=act_cfg, drop=drop)

        self.conv1x1_pw = nn.Sequential(
            nn.Conv2d(in_channels=2 * cn_dim, out_channels=dim, kernel_size=to_2tuple(1), stride=to_2tuple(1),
                      padding=0),
            nn.BatchNorm2d(dim)
        )

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, h, w):
        B, C, H, W = x.shape
        assert h == self.input_resolution[0] and w == self.input_resolution[1], "input feature has wrong size"

        x_shortcut = x
        x = self.conv1x1_bn_gelu(x)

        cn_x = x + self.act(self.conv3x3_dw(x))

        at_x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size[0], p2=self.patch_size[1])
        at_shortcut = at_x
        at_x = self.norm1(at_x)
        at_x = self.attn(at_x)
        at_x = at_shortcut + self.drop_path(at_x)
        at_x = at_x + self.drop_path(self.mlp(self.norm2(at_x)))
        at_x = rearrange(at_x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=self.patch_size[0],
                         p2=self.patch_size[1])

        cat_x = torch.cat([cn_x, at_x], 1)
        out = x_shortcut + self.conv1x1_pw(cat_x)

        return out


def IVT_backbone(num_classes):
    model = RMTransformer(img_size=(224, 224), in_chans=3, num_classes=num_classes)
    return model


if __name__ == '__main__':
    rmt = RMTransformer(num_class=600)
    x = torch.randn(size=(1, 3, 224, 224))
    out = rmt(x)
    # print(rmt)
    print(out.shape)
    pass
