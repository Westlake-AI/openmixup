from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_norm_layer,
                      constant_init, trunc_normal_init)
from mmcv.runner.base_module import BaseModule
from openmixup.models.backbones.vision_transformer import TransformerEncoderLayer

from .. import builder
from ..registry import NECKS
from ..utils import (build_2d_sincos_position_embedding,
                     CAETransformerRegressorLayer, trunc_normal_)


@NECKS.register_module()
class BEiTNeck(BaseModule):
    """Neck for BEiT Pre-training.

    Args:
        num_classes (int): The number of tokenized classes for the final prediction.
            Defaults to 8192.
        in_channels (int): The embed dims of latent feature in regressor and
            decoder. Defaults to 768.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes=8192,
                 in_channels=768,
                 init_cfg=None):
        super(BEiTNeck, self).__init__(init_cfg=init_cfg)

        self.decoders = nn.Linear(
            in_channels, num_classes) if num_classes > 0 else nn.Identity()

    def init_weights(self):
        if self.init_cfg is not None:
            super(BEiTNeck, self).init_weights()
            return
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generate the latent prediction and final prediction.

        Args:
            x (torch.Tensor): Features of tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final prediction.
        """
        logits = self.decoders(x)
        logits = logits.view(-1, logits.shape[-1])

        return logits


@NECKS.register_module()
class CAENeck(BaseModule):
    """Neck for CAE Pre-training.

    This module construct the latent prediction regressor and the decoder
    for the latent prediction and final prediction.

    Args:
        patch_size (int): The patch size of each token. Defaults to 16.
        num_classes (int): The number of classes for final prediction. Defaults
            to 8192.
        embed_dims (int): The embed dims of latent feature in regressor and
            decoder. Defaults to 768.
        regressor_depth (int): The number of regressor blocks. Defaults to 6.
        decoder_depth (int): The number of decoder blocks. Defaults to 8.
        num_heads (int): The number of head in multi-head attention. Defaults to 12.
        mlp_ratio (int): The expand ratio of latent features in MLP. defaults to 4.
        qkv_bias (bool): Whether or not to use qkv bias. Defaults to True.
        qk_scale (float, optional): The scale applied to the results of qk.
            Defaults to None.
        drop_rate (float): The dropout rate. Defaults to 0.
        attn_drop_rate (float): The dropout rate in attention block. Defaults to 0.
        norm_cfg (dict): The config of normalization layer. Defaults to
            dict(type='LN', eps=1e-6).
        init_values (float, optional): The init value of gamma. Defaults to None.
        mask_tokens_num (int): The number of mask tokens. Defaults to 75.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 patch_size=16,
                 num_classes=8192,
                 embed_dims=768,
                 regressor_depth=6,
                 decoder_depth=8,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_values=None,
                 mask_tokens_num=75,
                 init_cfg=None):
        super(CAENeck, self).__init__(init_cfg=init_cfg)

        self.num_features = self.embed_dim = embed_dims
        self.patch_size = patch_size
        self.mask_token_num = mask_tokens_num

        # regressor
        regressor_drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, regressor_depth)
        ]
        self.regressors = nn.ModuleList([
            CAETransformerRegressorLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=regressor_drop_path_rates[i],
                norm_cfg=norm_cfg,
                init_values=init_values) for i in range(regressor_depth)
        ])

        # decoder
        decoder_drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)
        ]

        self.decoders = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=decoder_drop_path_rates[i],
                norm_cfg=norm_cfg,
                init_values=init_values) for i in range(decoder_depth)
        ])

        _, self.norm_regressor = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        _, self.norm_decoder = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)

        self.head = nn.Linear(
            embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

    def init_weights(self) -> None:
        if self.init_cfg is not None:
            super(CAENeck, self).init_weights()
            return
        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.head.weight, std=0.02)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_unmasked: torch.Tensor, pos_embed_masked: torch.Tensor,
                pos_embed_unmasked: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generate the latent and final prediction of CAE.

        Args:
            x_unmasked (torch.Tensor): Features of unmasked tokens.
            pos_embed_masked (torch.Tensor): Position embedding of masked
                tokens.
            pos_embed_unmasked (torch.Tensor): Position embedding of unmasked
                tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final prediction and latent
                prediction.
        """
        x_masked = self.mask_token.expand(
            x_unmasked.shape[0], self.mask_token_num, -1)
        # regressor
        for regressor in self.regressors:
            x_masked = regressor(
                x_masked, torch.cat([x_unmasked, x_masked], dim=1),
                pos_embed_masked,
                torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1))
        x_masked = self.norm_regressor(x_masked)
        latent_pred = x_masked

        # decoder
        x_masked = x_masked + pos_embed_masked
        for decoder in self.decoders:
            x_masked = decoder(x_masked)
        x_masked = self.norm_decoder(x_masked)

        logits = self.head(x_masked)

        return (logits, latent_pred)


@NECKS.register_module()
class MAEPretrainDecoder(BaseModule):
    """Decoder for MAE Pre-training.

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

    Some of the code is borrowed from
    `https://github.com/facebookresearch/mae`.

    Example:
        >>> from mmselfsup.models import MAEPretrainDecoder
        >>> import torch
        >>> self = MAEPretrainDecoder()
        >>> self.eval()
        >>> inputs = torch.rand(1, 50, 1024)
        >>> ids_restore = torch.arange(0, 196).unsqueeze(0)
        >>> level_outputs = self.forward(inputs, ids_restore)
        >>> print(tuple(level_outputs.shape))
        (1, 196, 768)
    """

    def __init__(self,
                 num_patches=196,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(MAEPretrainDecoder, self).__init__(init_cfg)
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

        self.decoder_norm_name, decoder_norm = build_norm_layer(
            norm_cfg, decoder_embed_dim, postfix=1)
        self.add_module(self.decoder_norm_name, decoder_norm)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True)

    def init_weights(self):
        if self.init_cfg is not None:
            super(MAEPretrainDecoder, self).init_weights()
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)
        # initialize position embedding and mask token
        decoder_pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.decoder_pos_embed.shape[-1],
            cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())
        trunc_normal_(self.mask_token, std=0.02)

    @property
    def decoder_norm(self):
        return getattr(self, self.decoder_norm_name)

    def forward(self, x, ids_restore):
        if isinstance(x, list):
            x = x[-1]
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


@NECKS.register_module()
class SimMIMNeck(BaseModule):
    """Pre-train Neck For SimMIM.

    This neck reconstructs the original image from the shrunk feature map.

    Args:
        in_channels (int): Channel dimension of the feature map.
        encoder_stride (int): The total stride of the encoder.
    """

    def __init__(self, in_channels=128, encoder_stride=32, init_cfg=None):
        super(SimMIMNeck, self).__init__(init_cfg)
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=encoder_stride**2 * 3,
                kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_init(m, std=0.02, bias=0)

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        x = self.decoder(x)

        return x


@NECKS.register_module()
class NonLinearMIMNeck(BaseModule):
    """Non-linear Neck For MIM Pre-training.

    This neck reconstructs the target image from the shrunk feature map.

    Args:
        in_channels (int): Channel dimension of the feature map. It should
            be the decoder output channel if decoder_cfg is not None.
        in_chans (int): The channel of input image. Defaults to 3.
        encoder_stride (int): The total stride of the encoder.
        decoder_cfg (dict): Config dict for non-linear blocks. Defaults to None.
        act_cfg (dict): Whether to use an activation function. Defaults to None.
    """

    def __init__(self,
                 in_channels=128,
                 in_chans=3,
                 kernel_size=1,
                 encoder_stride=32,
                 decoder_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super(NonLinearMIMNeck, self).__init__(init_cfg)
        assert decoder_cfg is None or isinstance(decoder_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.decoder = builder.build_neck(decoder_cfg) \
            if decoder_cfg is not None else None
        self.activate = build_activation_layer(act_cfg) \
            if act_cfg is not None else None
        self.decoder_pred = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=encoder_stride**2 * in_chans,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.PixelShuffle(encoder_stride),
        )

    def init_weights(self):
        if self.init_cfg is not None:
            super(NonLinearMIMNeck, self).init_weights()
            return
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        assert isinstance(x, list)
        if self.decoder is not None:
            dec = self.decoder([x[-1]])[0]
        else:
            dec = x[-1]

        dec = self.decoder_pred(dec)
        if self.activate is not None:
            dec = self.activate(dec)

        return [dec]
