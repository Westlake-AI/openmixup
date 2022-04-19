# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
# modified from mmselfsup simmim_swin.py
import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init, trunc_normal_

from .swin_transformer import SwinTransformer
from ..builder import BACKBONES


@BACKBONES.register_module()
class SimMIMSwinTransformer(SwinTransformer):
    """Swin Transformer for SimMIM pre-training.

    Args:
        mask_layer (int): Layer to start MIM (mask img and add mask_token).
            Defaults to 0.
        mask_token (str): Mode of applying mask token in {None, 'randn', 'zero',
            'learnable', 'mean'}. Defaults to 'learnable'.
    """

    def __init__(self, mask_layer=0, mask_token='learnable', **kwargs):
        super().__init__(**kwargs)
        self.mask_layer = mask_layer
        self.mask_mode = mask_token
        assert self.mask_layer in [0, 1, 2, 3, 4,]
        assert self.mask_mode in [None, 'randn', 'zero', 'mean', 'learnable',]
        if self.mask_mode is not None:
            self.mask_token = nn.Parameter(
                torch.zeros(1, 1, self.embed_dims * (2 ** max(0, self.mask_layer-1))))

    def init_weights(self, pretrained=None):
        """Initialize weights."""
        super(SimMIMSwinTransformer, self).init_weights(pretrained)

        if pretrained is not None:
            # init pos embed
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            # init mask token
            if self.mask_mode != 'zero':
                trunc_normal_(self.mask_token, mean=0, std=.02)
            if self.mask_mode != 'learnable':
                self.mask_token.requires_grad = False
            
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0)

    def forward_mask(self, x, mask):
        """ perform MIM with mask and mask_token """
        if self.mask_mode is None:
            return x
        B, L, _ = x.shape
        if self.mask_mode == 'mean':
            self.mask_token.data = 0.999 * self.mask_token.data + \
                                   0.001 * x.mean(dim=[0, 1,], keepdim=True)
        mask_token = self.mask_token.expand(B, L, -1)
        mask = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1. - mask) + mask_token * mask
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
        x, hw_shape = self.patch_embed(x)
        
        if self.mask_layer == 0 and mask is not None:
            x = self.forward_mask(x, mask)
        
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)
        
        outs = []
        if -1 in self.out_indices:
            outs.append(
                x.view(x.size(0), *hw_shape, -1).permute(0, 3, 1, 2).contiguous())
        
        for i, stage in enumerate(self.stages):
            if self.mask_layer == i+1 and mask is not None:
                x = self.forward_mask(x, mask)
            
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                        stage.out_channels).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs