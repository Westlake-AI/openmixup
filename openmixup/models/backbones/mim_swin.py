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
        Args:
        arch (str | dict): Swin Transformer architecture
            Defaults to 'T'.
        img_size (int | tuple): The size of input image.
            Defaults to 224.
        in_channels (int): The num of input channels.
            Defaults to 3.
        drop_rate (float): Dropout rate after embedding.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate.
            Defaults to 0.1.
        out_indices (tuple): Layers to be outputted. Defaults to (3, ).
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer at end
            of backone. Defaults to dict(type='LN')
        stage_cfgs (Sequence | dict): Extra config dict for each
            stage. Defaults to empty dict.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to empty dict.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

    def init_weights(self, pretrained=None):
        """Initialize weights."""
        super(SwinTransformer, self).init_weights(pretrained)

        if pretrained is not None:
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)

            trunc_normal_(self.mask_token, mean=0, std=.02)

            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0)

    def forward(self, x, mask):
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

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1. - w) + mask_token * w

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed

        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                        stage.out_channels).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs
