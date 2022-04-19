import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.runner.base_module import BaseModule
from openmixup.models.utils.weight_init import trunc_normal_
from openmixup.models.backbones.vision_transformer import TransformerEncoderLayer

from .. import builder
from ..registry import NECKS
from ..utils import build_2d_sincos_position_embedding


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
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(MAEPretrainDecoder, self).__init__()
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

    def __init__(self, in_channels=128, encoder_stride=32):
        super(SimMIMNeck, self).__init__()
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
        norm_token (None or str): Mode of applying denormalization before the
            decoder_pred. Defaults to False.
    """

    def __init__(self,
                 in_channels=128,
                 in_chans=3,
                 kernel_size=1,
                 encoder_stride=32,
                 decoder_cfg=None,
                 norm_token=None,
                ):
        super(NonLinearMIMNeck, self).__init__()
        assert decoder_cfg is None or isinstance(decoder_cfg, dict)
        self.decoder = builder.build_neck(decoder_cfg) \
            if decoder_cfg is not None else None
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
        self.norm_mode = norm_token
        assert self.norm_mode in [None, 'AdaLN', 'AdaIN',]
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)

    @staticmethod
    def _calc_instance_norm(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_std = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_std.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    @staticmethod
    def _calc_layer_norm(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N = size[0]
        feat_std = feat.var(dim=[1, 2, 3]) + eps
        feat_std = feat_std.sqrt().view(N, 1, 1, 1)
        feat_mean = feat.mean(dim=[1, 2, 3]).view(N, 1, 1, 1)
        return feat_mean, feat_std
    
    def forward(self, x):
        assert isinstance(x, list)

        if self.decoder is not None:
            dec = self.decoder([x[-1]])[0]
        else:
            dec = x[-1]
        
        outs = []
        if self.norm_mode is not None:
            assert len(x) >= 2 and (x[0].size()[:2] == dec.size()[:2])
            size = dec.size()
            if self.norm_mode == 'AdaIN':
                feat_mean, feat_std = self._calc_instance_norm(x[0].detach())
                content_mean, content_std = self._calc_instance_norm(dec)
                dec = (dec - content_mean.expand(size)) / content_std.expand(size)
            elif self.norm_mode == 'AdaLN':
                feat_mean, feat_std = self._calc_layer_norm(x[0].detach())
                content_mean, content_std = self._calc_layer_norm(dec)
                dec = (dec - content_mean.expand(size)) / content_std.expand(size)
            
            dec = dec * feat_mean.expand(size) + feat_std.expand(size)
            outs.append(dec)

        dec = self.decoder_pred(dec)
        outs.append(dec)
        
        return outs
