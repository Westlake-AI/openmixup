import torch

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init
from mmcv.runner.base_module import BaseModule

import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from torchvision import transforms


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_padding(img_size=(224, 112, 56), patch_size=16, num_sizes=3):
    img_size = sorted(img_size, reverse=True)
    num_patch = img_size[0] // patch_size
    opera = num_patch - 1

    somlist = []
    for i in range(0, patch_size + 1):
        somlist.append(opera * i)

    padding_list = []
    stride_list = []
    for i in range(num_sizes):
        some = img_size[i] - patch_size
        idx = find_nearest(somlist, some)
        if some in somlist:
            padding_list.append(0)

        else:
            if somlist[idx] < some:
                idx = idx + 1
            padding_list.append(math.ceil((abs(somlist[idx] - some)) / 2))

        stride_list.append(idx)

    return img_size, stride_list, padding_list, num_patch

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

class Attention(BaseModule):
    def __init__(self,
                dim,
                num_patch=None,
                num_heads=8,
                qkv_bias=False,
                qk_scale=None,
                attn_drop=0.,
                proj_drop=0.,
                with_qkv=True,
                init_cfg=None):
        super(Attention, self).__init__(init_cfg=init_cfg)
        """
        dim=768, num_heads=12, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.num_patch = num_patch

        if num_patch is not None:
            self.window_size = num_patch
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * num_patch[0] - 1) * (2 * num_patch[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.num_patch is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(BaseModule):
    def __init__(self,
                dim,
                num_patch,
                num_heads,
                mlp_ratio=4.,
                qkv_bias=False,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.1,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                attention_type='multi_scale_correspondence_union',
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                init_cfg=None):
        super(Block, self).__init__(init_cfg)

        self.attention_type = attention_type
        assert (attention_type in ['multi_scale_correspondence_union', 'multi_scale_full_union'])

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, dim, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = Attention(
            dim, num_patch=None, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)

        if self.attention_type == 'multi_scale_correspondence_union':
            # self.Multi_scale_norm1 = norm_layer(dim)
            self.Multi_scale_norm1_name, Multi_scale_norm1 = build_norm_layer(
                norm_cfg, dim, postfix=1)
            self.add_module(self.Multi_scale_norm1_name, Multi_scale_norm1)
            self.Multi_scale_attn = Attention(
                dim, num_patch=None, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop)
            # self.Multi_scale_norm2 = norm_layer(dim)
            self.Multi_scale_norm2_name, Multi_scale_norm2 = build_norm_layer(
                norm_cfg, dim, postfix=1)
            self.add_module(self.Multi_scale_norm2_name, Multi_scale_norm2)
            self.Multi_scale_fc = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, dim, postfix=1)
        self.add_module(self.norm2_name, norm2)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_cfg=act_cfg, drop=drop)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def Multi_scale_norm1(self):
        return getattr(self, self.norm2_name)

    @property
    def Multi_scale_norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, B, S, H, W): 

        if self.attention_type in ['multi_scale_full_union']:

            x = x + self.drop_path(self.attn(self.norm1(x))) 
            x = x + self.drop_path(self.mlp(self.norm2(x))) 
            return x
        elif self.attention_type == 'multi_scale_correspondence_union':

            # Corresponding scale
            res_scale = x
            x = rearrange(x, 'b (h w s) m -> (b h w) s m', b=B, h=H, w=W, s=S) 
            x = self.drop_path(
                self.Multi_scale_attn(self.Multi_scale_norm1(x)))
            x = rearrange(x, '(b h w) s m -> b (h w s) m', b=B, h=H, w=W, s=S)
            x = self.Multi_scale_fc(self.Multi_scale_norm2(x)) 
            x = res_scale + x 

            # Spatial
            x_spatial = x
            x = rearrange(x, 'b (h w s) m -> (b s) (h w) m', b=B, h=H, w=W,
                          s=S) 
            x = self.drop_path(self.attn(self.norm1(x))) 
            x = rearrange(x, '(b s) (h w) m -> b (h w s) m', b=B, h=H, w=W,
                          s=S) 

            x = x_spatial + x 
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=(224, 112, 56), patch_size=16, in_chans=3, embed_dim=768, num_sizes=3):

        super().__init__()
        img_size, stride_list, padding_list, num_patch = calculate_padding(img_size=img_size, patch_size=patch_size,
                                                                           num_sizes=num_sizes)
        self.num_sizes = num_sizes
        self.num_patch = num_patch

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patch * num_patch

        self.modlist = nn.ModuleList()

        for i in range(num_sizes):
            self.modlist.append(nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_list[i],
                          padding=padding_list[i])
            ))

        # self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, feature_list):
        S = len(feature_list)

        for i in range(self.num_sizes):
            B, C, H, W = feature_list[i].shape

            x = rearrange(feature_list[i], 'b c h w -> (b) c h w')

            x = self.modlist[i](x)  # 'b embed_dim num_patch num_patch''1 768 14 14'

            # W = x.size(-1)  # 14

            x = x.flatten(2).transpose(1, 2)  # 'b num_patch * num_patch embed_dim''1 14*14 768'

            feature_list[i] = x

        x = torch.stack(feature_list, 1)  # '1 3 14*14 768'

        x = rearrange(x, 'b s n f -> (b s) n f')  # '3 14*14 768'

        return x, S, self.num_patch


class PatchConvNN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.PatchConvblock = GranularStack(in_c=dim)

    def forward(self, x):
        x = self.PatchConvblock(x)
        return x


class GranularStack(nn.Module):
    def __init__(self, in_c, output_layer=False):
        super(GranularStack, self).__init__()
        self.Stage1 = GranularStackBlock(in_c=in_c)
        self.Stage2 = GranularStackBlock(in_c=2 * in_c)

    def forward(self, x):
        x = self.Stage1(x)
        x = self.Stage2(x)
        return x


class GranularStackBlock(nn.Module):
    def __init__(self, in_c):
        super(GranularStackBlock, self).__init__()
        embed = in_c // 2
        self.branch1_conv1 = nn.Conv2d(in_channels=in_c, out_channels=embed, kernel_size=1)
        self.branch1_bn_conv1 = nn.BatchNorm2d(embed)
        self.gelu = nn.GELU()

        self.branch1_conv2 = nn.Conv2d(in_channels=embed, out_channels=embed, kernel_size=1)
        self.branch1_bn_conv2 = nn.BatchNorm2d(embed)

        self.branch3_conv1 = nn.Conv2d(in_channels=in_c, out_channels=embed, kernel_size=1)
        self.branch3_bn_conv1 = nn.BatchNorm2d(embed)

        self.branch3_conv2 = nn.Conv2d(in_channels=embed, out_channels=embed, kernel_size=3, stride=1, padding=1,
                                       groups=embed)
        self.branch3_bn_conv2 = nn.BatchNorm2d(embed)

        self.branch3_conv3 = nn.Conv2d(in_channels=embed, out_channels=embed, kernel_size=1)
        self.branch3_bn_conv3 = nn.BatchNorm2d(embed)

    def forward(self, x):
        x1 = self.branch1_conv1(x)
        x1 = self.branch1_bn_conv1(x1)
        x1 = self.gelu(x1)
        x1 = self.branch1_conv2(x1)
        x1 = self.branch1_bn_conv2(x1)
        x1 = self.gelu(x1)

        x3 = self.branch3_conv1(x)
        x3 = self.branch3_bn_conv1(x3)
        x3 = self.gelu(x3)
        x3 = self.branch3_conv2(x3)
        x3 = self.branch3_bn_conv2(x3)
        x3 = self.gelu(x3)
        x3 = self.branch3_conv3(x3)
        x3 = self.branch3_bn_conv3(x3)
        x3 = self.gelu(x3)

        x = torch.cat((x1, x, x3), dim=1)
        return x


class DownSampleLayer(nn.Module):

    def __init__(self, dim_in, dim_out, downsample_rate):
        super().__init__()
        self.downsample = nn.Conv2d(dim_in, dim_out, kernel_size=downsample_rate, stride=downsample_rate)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.downsample(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class BasicLayer(BaseModule):
    def __init__(self,
                dim,
                depth,
                num_patch,
                num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                downsample=None,
                downsample_rate=2, 
                attention_type='multi_scale_correspondence_union',
                use_checkpoint=False,
                init_cfg=None):
        super(BasicLayer, self).__init__(init_cfg=None)
        self.dim = dim
        self.depth = depth
        self.attention_type = attention_type
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_patch=to_2tuple(num_patch), num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_cfg=act_cfg, norm_cfg=norm_cfg, attention_type=self.attention_type)
            for i in range(self.depth)])

        self.Conv2 = PatchConvNN(dim=dim)
        self.downsample = downsample(dim_in=4 * dim, dim_out=2 * dim, downsample_rate=downsample_rate)

    def forward(self, x, B, S, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                pass
            else:
                x = blk(x, B, S, H, W)
        x = rearrange(x, 'b (h w s) m -> (b s) m h w', b=B, h=H, w=W, s=S)
        x = self.Conv2(x)
        x = self.downsample(x)

        return x


class MultiScaleUnionTransformer(BaseModule):
    """Label Enhancement-Based Multiscale Transformer for Palm-Vein Recognition.

    A PyTorch implement of : `Label Enhancement-Based Multiscale Transformer for 
    Palm-Vein Recognition <https://ieeexplore.ieee.org/document/10081428>`_
    
    Published on IEEE Transactions on Instrumentation and Measurement 2023

    Modified by Xin Jin`_

    """
    def __init__(self,
                img_size=(224, 96, 64),
                patch_size=4,
                in_chans=3,
                num_classes=600,
                embed_dim=128,
                downsample_rate=(2, 2, 2, 2),
                depths=(1, 1, 1, 1),
                num_heads=(4, 8, 16, 32),
                mlp_ratio=4.,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                # norm_layer=nn.LayerNorm,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                num_sizes=3,
                attention_type='multi_scale_correspondence_union',
                dropout=0.,
                use_checkpoint=False,
                init_cfg=None,
                **kwargs):
        super(MultiScaleUnionTransformer, self).__init__(init_cfg)
        self.attention_type = attention_type
        self.num_layers = len(depths)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_sizes = num_sizes
        self.num_features = embed_dim
        self.embed_dim = embed_dim * 2 ** self.num_layers
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_sizes=num_sizes)
        num_patches = self.patch_embed.num_patches  # '14*14'
        num_patch = self.patch_embed.num_patch
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_embed = nn.Parameter(torch.zeros(1, 1 + num_sizes, embed_dim))
        self.scale_drop = nn.Dropout(p=drop_rate)
        trunc_normal_(self.scale_embed, std=.02)
        trunc_normal_(self.scale_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule随机深度衰减规则

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_patch=int(num_patch * 2 ** -i_layer),
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=DownSampleLayer,
                               downsample_rate=downsample_rate[i_layer],
                               act_cfg=act_cfg,
                               norm_cfg=norm_cfg,
                               use_checkpoint=use_checkpoint,
                               attention_type=self.attention_type)
            self.layers.append(layer)

        self.apply(self.init_weights)

        if self.attention_type == 'multi_scale_correspondence_union':
            for j in range(0, self.num_layers):
                i = 0
                for m in self.layers[j].blocks.modules():
                    m_str = str(m)
                    if 'Block' in m_str:
                        if i > 0:
                            nn.init.constant_(m.Multi_scale_fc.weight, 0)
                            nn.init.constant_(m.Multi_scale_fc.bias, 0)
                        i += 1

        self.img_transforms = []
        for i in img_size:
            self.img_transforms.append(transforms.Resize((i, i), antialias=True))

    def init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias.data, 0)
            nn.init.normal_(m.weight.data, 1.0, 0.02)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'scale_embed', 'scale_token'}

    def forward_features(self, x): 
        feature_list = [i(x) for i in self.img_transforms]
        B = feature_list[0].shape[0] 
        x, S, W = self.patch_embed(feature_list) 

        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed 
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2) 
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W  
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P) 
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed 
        x = self.pos_drop(x) 

        x = rearrange(x, '(b s) n m -> (b n) s m', b=B, s=S)
        scale_tokens = self.scale_token.expand(x.size(0), -1, -1) 
        x = torch.cat((scale_tokens, x), dim=1) 
        # Resizing Scale embeddings in case they don't match
        if 1 + S != self.scale_embed.size(1):
            scale_embed = self.scale_embed.transpose(1, 2) 
            new_scale_embed = F.interpolate(scale_embed, size=(1 + S), mode='nearest')
            new_scale_embed = new_scale_embed.transpose(1, 2) 
            x = x + new_scale_embed
        else:
            x = x + self.scale_embed
        x = self.scale_drop(x)
        x = rearrange(x, '(b n) s m -> b (n s) m', b=B, s=1 + S)  

        S = 1 + S
        num_spatial_tokens = x.size(1) // S 

        H = num_spatial_tokens // W 
        # Attention blocks
        for i in range(0, self.num_layers):
            x = self.layers[i](x, B, S, H, W)
            if i < self.num_layers - 1:
                H = x.size(-2)
                W = x.size(-1)
                x = rearrange(x, '(b s) m h w -> b (h w s) m', b=B, h=H, w=W, s=S)
        return x

    def forward(self, x):
        out = self.forward_features(x)  
        return out


def MSVT_backbone(num_classes):
    if num_classes == 600:
        model = MultiScaleUnionTransformer(num_classes=600)
    else:
        raise NotImplementedError(f"num_classes={num_classes}")
    return model


if __name__ == '__main__':
    rmt = MSVT_backbone(num_classes=600)
    x = torch.randn(size=(1, 3, 224, 224))
    out = rmt(x)
    print(out.shape)
    pass