import math
import numpy as np
import torch
from torch import nn
from mmcv.cnn import build_norm_layer, xavier_init, constant_init

from ..builder import BACKBONES
from .vision_transformer import TransformerEncoderLayer, VisionTransformer
from ..utils import (build_2d_sincos_position_embedding,
                     RelativePositionBias, trunc_normal_)


@BACKBONES.register_module()
class MAEViT(VisionTransformer):
    """Vision Transformer for MAE pre-training.

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 mask_ratio=0.75,
                 init_cfg=None,
                 **kwargs):
        super().__init__(arch=arch,
                         img_size=img_size,
                         patch_size=patch_size,
                         in_channels=in_channels,
                         out_indices=out_indices,
                         drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate,
                         qkv_bias=qkv_bias,
                         norm_cfg=norm_cfg,
                         final_norm=final_norm,
                         with_cls_token=with_cls_token,
                         output_cls_token=output_cls_token,
                         interpolate_mode=interpolate_mode,
                         patch_cfg=patch_cfg,
                         layer_cfgs=layer_cfgs,
                         init_cfg=init_cfg,
                         **kwargs)

        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def init_weights(self, pretrained=None):
        super(MAEViT, self).init_weights(pretrained)

        if pretrained is None:
            # initialize position embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5),
                self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())
            w = self.patch_embed.projection.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            trunc_normal_(self.cls_token, std=0.02, bias=0)

            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_init(m, gain=1, bias=0, distribution='normal')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            constant_init(m, val=1, bias=0)

    def random_masking(self, x, mask_ratio=0.75):
        """Generate the mask for MAE Pre-training.

        Args:
            x (torch.tensor): Image with data augmentation applied.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            tuple[Tensor, Tensor, Tensor]: masked image, mask and the ids
                to restore original image.

            - x_masked (Tensor): masked image.
            - mask (Tensor): mask used to mask image.
            - ids_restore (Tensor): ids to restore original image.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        """ MAE backbone only used for MAE model """
        B = x.shape[0]
        x, _ = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return (x, mask, ids_restore)


@BACKBONES.register_module()
class MIMVisionTransformer(VisionTransformer):
    """Vision Transformer for MIM-style model (Mask Image Modeling)
    classification (fine-tuning or linear probe).

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        finetune (bool): Whether or not do fine-tuning. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 use_window=False,
                 drop_rate=0,
                 drop_path_rate=0,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 init_values=0.0,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 finetune=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(arch,
                         img_size=img_size,
                         patch_size=patch_size,
                         out_indices=out_indices,
                         use_window=use_window,
                         drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate,
                         qkv_bias=qkv_bias,
                         norm_cfg=norm_cfg,
                         final_norm=final_norm,
                         output_cls_token=output_cls_token,
                         interpolate_mode=interpolate_mode,
                         init_values=init_values,
                         patch_cfg=patch_cfg,
                         layer_cfgs=layer_cfgs,
                         init_cfg=init_cfg,
                         **kwargs)
        if not self.final_norm:
            _, self.fc_norm = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)

        self.finetune = finetune
        if not self.finetune:
            self.frozen_stages = self.num_layers - 1  # all layers
            self._freeze_stages()

    def forward(self, x):
        B = x.shape[0]
        x, _ = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        if not self.final_norm:
            x = x[:, 1:, :].mean(dim=1)
            outs = self.fc_norm(x)
        else:
            outs = x[:, 0]
        return [outs]


@BACKBONES.register_module()
class SimMIMViT(VisionTransformer):
    """Vision Transformer for SimMIM pre-training.

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        mask_layer (int): Layer to start MIM (mask img and add mask_token).
            Defaults to 0.
        mask_token (str): Mode of applying mask token in {None, 'randn', 'zero',
            'learnable', 'mean'}. Defaults to 'learnable'.
        mask_init (float): The init values of mask_token gamma. Defaults to 0.0.
    """

    def __init__(self,
                 mask_layer=0,
                 mask_token='learnable',
                 mask_init=0,
                 replace=True,
                 detach=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_layer = mask_layer
        self.mask_mode = mask_token
        self.replace = replace
        self.detach = detach
        assert 0 <= self.mask_layer < self.num_layers
        assert self.mask_mode in [None, 'randn', 'zero', 'mean', 'learnable',]
        if self.mask_mode is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        if mask_init > 0 and not replace:
            self.mask_gamma = nn.Parameter(
                mask_init * torch.ones((self.embed_dims)), requires_grad=True)
        else:
            self.mask_gamma = None

    def init_weights(self, pretrained=None):
        super(SimMIMViT, self).init_weights(pretrained)

        if pretrained is None:
            # init mask token
            if self.mask_mode is not None:
                if self.mask_mode != 'zero':
                    trunc_normal_(self.mask_token, std=0.02, bias=0)
                if self.mask_mode != 'learnable':
                    self.mask_token.requires_grad = False

    def forward_mask(self, x, mask=None):
        """ perform MIM with mask and mask_token """
        if self.mask_mode is None:
            return x
        assert mask is not None
        B, L, _ = x.shape
        if self.mask_mode == 'mean':
            self.mask_token.data = x.mean(dim=[0, 1,], keepdim=True)
        mask = mask.flatten(1).unsqueeze(-1).type_as(x)

        if mask.size(1) + 1 == L:  # with cls_token
            mask_token = self.mask_token.expand(B, L-1, -1)
            if self.replace:
                x[:, 1:] = x[:, 1:] * (1. - mask) + mask_token * mask
            else:
                if self.detach:
                    x[:, 1:] = x[:, 1:] * (1. - mask) + x[:, 1:].clone().detach() * mask
                if self.mask_gamma is not None:
                    x[:, 1:] = x[:, 1:] * (1. - mask) + (x[:, 1:] * mask) * self.mask_gamma
                x[:, 1:] = x[:, 1:] + mask_token * mask
        elif mask.size(1) == L:
            mask_token = self.mask_token.expand(B, L, -1)
            if self.replace:
                x = x * (1. - mask) + mask_token * mask
            else:
                if self.detach:
                    x = x * (1. - mask) + x.clone().detach() * mask
                if self.mask_gamma is not None:
                    x = x * (1. - mask) + (x * mask) * self.mask_gamma
                x = x + mask_token * mask  # residual
        else:
            raise NotImplementedError
        return x

    def forward(self, x, mask=None):
        """Generate features for masked images.

        This function generates mask images and get the hidden features for
        them.

        Args:
            x (torch.Tensor): Input images.
            mask (torch.Tensor): Masks used to construct masked images.

        Returns:
            tuple: A tuple containing features from multi-stages.
        """
        x, _ = self.patch_embed(x)

        if self.mask_layer == 0:
            x = self.forward_mask(x, mask)
        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            if self.mask_layer == i+1:
                x = self.forward_mask(x, mask)
            
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)
            
            if i in self.out_indices:
                if self.with_cls_token:
                    x = x[:, 1:]
                B, L, C = x.shape
                H = W = int(L ** 0.5)
                x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
                outs.append(x)
        
        return outs


@BACKBONES.register_module()
class BEiTViT(VisionTransformer):
    """Vision Transformer for BEiT pre-training.

    Rewritten version of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed. Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        use_shared_rel_pos_bias (bool): Whether or not use shared relative
            position bias. Defaults to True.
        init_values (float): The initialization value for
            the learnable scaling of attention and FFN. Defaults to 0.1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 output_cls_token=True,
                 frozen_stages=-1,
                 use_shared_rel_pos_bias=True,
                 init_values=0,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(padding=0),
                 layer_cfgs=dict(),
                 init_cfg=None,
                 **kwargs):
        super().__init__(arch=arch,
                         img_size=img_size,
                         patch_size=patch_size,
                         in_channels=in_channels,
                         out_indices=out_indices,
                         drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate,
                         qkv_bias=qkv_bias,
                         norm_cfg=norm_cfg,
                         final_norm=final_norm,
                         with_cls_token=with_cls_token,
                         output_cls_token=output_cls_token,
                         frozen_stages=frozen_stages,
                         init_values=init_values,
                         interpolate_mode=interpolate_mode,
                         patch_cfg=patch_cfg,
                         layer_cfgs=layer_cfgs,
                         init_cfg=init_cfg,
                         **kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        if use_shared_rel_pos_bias:
            self.shared_rel_pos_bias = RelativePositionBias(
                window_size=self.patch_resolution,
                num_heads=self.arch_settings['num_heads'])
        else:
            self.shared_rel_pos_bias = None

    def init_weights(self, pretrained=None):
        """Initialize position embedding, patch embedding and cls token."""
        super(BEiTViT, self).init_weights(pretrained)

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:  # None
                trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.mask_token, std=0.02)
            self.apply(self._init_weights)
            self.fix_init_weight()

    def fix_init_weight(self) -> None:

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # replace the masked visual tokens by mask_token
        L = x.shape[1]
        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1. - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            x = x[:, 1:]

        shared_rel_pos_bias = self.shared_rel_pos_bias() \
            if self.shared_rel_pos_bias is not None else None
        for i, layer in enumerate(self.layers):
            if self.shared_rel_pos_bias is not None:
                x = layer(x, rel_pos_bias=shared_rel_pos_bias)
            else:
                x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return x


@BACKBONES.register_module()
class CAEViT(VisionTransformer):
    """Vision Transformer for CAE pre-training.

    Rewritten version of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        init_values (float, optional): The init value of gamma in
            TransformerEncoderLayer.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 init_values=None,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None,
                 **kwargs):
        super().__init__(arch=arch,
                         img_size=img_size,
                         patch_size=patch_size,
                         out_indices=out_indices,
                         drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate,
                         norm_cfg=norm_cfg,
                         final_norm=final_norm,
                         output_cls_token=output_cls_token,
                         interpolate_mode=interpolate_mode,
                         patch_cfg=patch_cfg,
                         layer_cfgs=layer_cfgs,
                         init_cfg=init_cfg,
                         **kwargs)
        self.pos_embed.requires_grad = False
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        # Replace original TransformerEncoderLayer with customized
        # TransformerEncoderLayer
        self.layers = nn.ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                init_values=init_values,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

    def init_weights(self, pretrained=None):
        super(CAEViT, self).init_weights(pretrained)

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # initialize position  embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5),
                self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())

            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, _ = self.patch_embed(img)
        batch_size, _, dim = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # NOTE: unmasked embeddings
        x_unmasked = x[~mask].reshape(batch_size, -1, dim)
        x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=1)

        pos_embed = self.pos_embed.expand(batch_size, self.num_patches + 1,
                                          dim)
        pos_embed_unmasked = pos_embed[:,
                                       1:][~mask].reshape(batch_size, -1, dim)
        pos_embed_unmasked = torch.cat((pos_embed[:, :1], pos_embed_unmasked),
                                       dim=1)
        x_unmasked = x_unmasked + pos_embed_unmasked

        x_unmasked = self.drop_after_pos(x_unmasked)

        for i, layer in enumerate(self.layers):
            x_unmasked = layer(x_unmasked)

            if i == len(self.layers) - 1 and self.final_norm:
                x_unmasked = self.norm1(x_unmasked)

        return x_unmasked
