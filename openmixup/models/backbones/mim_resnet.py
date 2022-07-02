import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..registry import BACKBONES
from .resnet_mmcls import ResNet


@BACKBONES.register_module()
class MIMResNet(ResNet):
    """ResNet backbone for MIM pre-training.

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
        super(MIMResNet, self).__init__(**kwargs)
        self.mask_layer = mask_layer
        self.mask_mode = mask_token
        self.replace = replace
        self.detach = detach
        assert self.mask_layer in [0, 1, 2, 3]
        assert self.mask_mode in [
            None, 'randn', 'zero', 'mean', 'instance_mean', 'learnable',]
        ARCH_DIMS = {
            **dict.fromkeys(
                ['18', '34'],
                [64, 128, 256, 512,]),
            **dict.fromkeys(
                ['50', '101', '152',],
                [64, 256, 512, 1024,]),
        }
        self.mask_dims = ARCH_DIMS[str(self.depth)][self.mask_layer]
        if self.mask_mode not in [None, 'instance_mean',]:
            self.mask_token = nn.Parameter(torch.zeros(1, self.mask_dims, 1, 1))
        if mask_init > 0 and not replace:
            self.mask_gamma = nn.Parameter(
                mask_init * torch.ones((1, self.mask_dims, 1, 1)), requires_grad=True)
        else:
            self.mask_gamma = None

    def init_weights(self, pretrained=None):
        """Initialize weights."""
        super(MIMResNet, self).init_weights(pretrained)

        if pretrained is not None:
            if self.mask_mode not in [None, 'instance_mean',]:
                if self.mask_mode != 'zero':
                    trunc_normal_(self.mask_token, mean=0, std=.02)
                if self.mask_mode != 'learnable':
                    self.mask_token.requires_grad = False

    def forward_mask(self, x, mask=None):
        """ perform MIM with mask and mask_token """
        B, _, H, W = x.size()
        if self.mask_mode is None:
            return x
        elif self.mask_mode == 'instance_mean':
            mask_token = x.mean(dim=[2, 3], keepdim=True).expand(B, -1, H, W)
        else:
            if self.mask_mode == 'mean':
                self.mask_token.data = x.mean(dim=[0, 2, 3], keepdim=True)
            mask_token = self.mask_token.expand(B, -1, H, W)
        assert mask is not None
        mask = mask.view(B, 1, H, W).type_as(mask_token)
        if self.replace:
            x = x * (1. - mask) + mask_token * mask
        else:
            if self.detach:
                x = x * (1. - mask) + x.clone().detach() * mask
            if self.mask_gamma is not None:
                x = x * (1. - mask) + (x * mask) * self.mask_gamma
            x = x + mask_token * mask  # residual
        return x

    def forward(self, x, mask=None):
        # stem
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.relu(self.norm1(self.conv1(x)))
        x = self.maxpool(x)
        
        outs = []
        if -1 in self.out_indices:
            outs.append(x)
        
        # stages
        for i, layer_name in enumerate(self.res_layers):
            # mask, add mask token
            if self.mask_layer == i:
                x = self.forward_mask(x, mask)
            
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs
        return outs
