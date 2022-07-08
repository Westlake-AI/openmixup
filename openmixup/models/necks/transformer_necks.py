import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.runner.base_module import BaseModule
from openmixup.models.backbones.vision_transformer import TransformerEncoderLayer

from ..registry import NECKS
from ..utils import build_2d_sincos_position_embedding, trunc_normal_


@NECKS.register_module()
class TransformerNeck(BaseModule):
    """Transformer Neck.

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 1024.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.
    """

    def __init__(self,
                 num_patches=196,
                 in_channels=1024,
                 embed_dims=512,
                 depth=8,
                 num_heads=16,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 with_cls_token=False,
                 with_mask_token=False,
                 init_cfg=None):
        super(TransformerNeck, self).__init__(init_cfg)

        self.num_patches = num_patches
        self.embed_dims = embed_dims
        self.neck_embed = nn.Linear(in_channels, embed_dims, bias=True)
        # fixed pos embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dims), requires_grad=False)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set cls token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.with_cls_token = with_cls_token
        # set mask token
        if with_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.with_mask_token = with_mask_token

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=int(mlp_ratio * embed_dims),
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(depth)
        ])

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

    def init_weights(self):
        if self.init_cfg is not None:
            super(TransformerNeck, self).init_weights()
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)
        # initialize position embedding and mask token
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.pos_embed.shape[-1],
            cls_token=True)
        self.pos_embed.data.copy_(pos_embed.float())
        # init tokens
        if self.with_cls_token:
            trunc_normal_(self.cls_token, mean=0, std=.02)
        if self.with_mask_token:
            trunc_normal_(self.mask_token, mean=0, std=.02)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        # embed tokens
        x = self.neck_embed(x)

        # apply tokens
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.with_mask_token:
            assert mask is not None
            mask_tokens = self.mask_token.expand(B, L, -1)
            mask = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1. - mask) + mask_tokens * mask
        
        # add pos embed
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm1(x)

        return x
