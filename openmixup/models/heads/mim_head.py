import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from torch.nn import functional as F
from mmcv.cnn.utils.weight_init import trunc_normal_init

from ..builder import build_loss
from ..registry import HEADS
from .cls_head import ClsHead
from openmixup.utils import print_log

try:
    import torch.fft as fft
except ImportError:
    fft = None


@HEADS.register_module
class MAEPretrainHead(BaseModule):
    """Pre-training head for MAE.

    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self, norm_pix=False, patch_size=16, init_cfg=None):
        super(MAEPretrainHead, self).__init__(init_cfg)
        self.norm_pix = norm_pix
        self.patch_size = patch_size

    def patchify(self, imgs):
        """
        Args:
            x (torch.Tensor): The shape is (N, L, patch_size**2 * 3)
        Returns:
            imgs (torch.Tensor): The shape is (N, 3, H, W)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        Args:
            x (torch.Tensor): The shape is (N, L, patch_size**2 *3)
        Returns:
            imgs (torch.Tensor): The shape is (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x, x_rec, mask):
        losses = dict()
        target = self.patchify(x)
        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (x_rec - target)**2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        losses['loss'] = loss
        return losses


@HEADS.register_module()
class MAEFinetuneHead(ClsHead):
    """Fine-tuning head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, **kwargs):
        super(MAEFinetuneHead, self).__init__(**kwargs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=2e-5, bias=0)

    def forward(self, x):
        """"Get the logits."""
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        return [self.fc(x)]


@HEADS.register_module()
class MAELinprobeHead(ClsHead):
    """Linear probing head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, in_channels=786, **kwargs):
        super(MAELinprobeHead, self).__init__(in_channels=in_channels, **kwargs)
        self.bn = nn.BatchNorm1d(in_channels, affine=False, eps=1e-6)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.01, bias=0)

    def forward(self, x):
        """"Get the logits."""
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = self.bn(x[0])
        return [self.fc(x)]


@HEADS.register_module
class SimMIMHead(BaseModule):
    """Pretrain Head for SimMIM.

    Args:
        encoder_in_channels (int): Number of input channels for encoder.
    """

    def __init__(self, encoder_in_channels=3, init_cfg=None):
        super(SimMIMHead, self).__init__(init_cfg)
        self.encoder_in_channels = encoder_in_channels

    def forward(self, x, x_rec, mask):
        scale_h, scale_w = x.size(2) / mask.size(1), x.size(3) / mask.size(2)
        if scale_h > 1:
            mask = mask.repeat_interleave(int(scale_h), 1).repeat_interleave(
                int(scale_w), 2).unsqueeze(1).contiguous()
        else:
            mask = F.interpolate(mask.type_as(x).unsqueeze(1),
                                 scale_factor=(scale_h, scale_w), mode="nearest")
        
        loss_rec = F.l1_loss(x_rec, x, reduction='none')
        loss = (loss_rec * mask).sum() / (mask.sum() +
                                          1e-5) / self.encoder_in_channels
        losses = dict()
        losses['loss'] = loss

        return losses


@HEADS.register_module
class A2MIMHead(BaseModule):
    """Head for A2MIM training.

    Args:
        loss (dict): Config of regression loss.
        encoder_in_channels (int): Number of input channels for encoder.
        unmask_weight (float): Loss weight for unmasked patches.
        fft_weight (float): Loss weight for the fft prediction loss. Default to 0.
        fft_reweight (bool): Whether to use the fft reweight loss. Default to False.
        fft_focal (bool): Whether to adopt the focal fft loss. Default to False.
        fft_unmask_replace (str): Mode to replace (detach) unmask patches for the fft
            loss, in {None, 'target', 'prediction', 'mean', 'mixed',}.
        fft_unmask_weight (float): Loss weight to caculate the fft loss on unmask
            tokens. Default to 0.
    """

    def __init__(self,
                 loss=dict(
                    type='RegressionLoss', loss_weight=1.0, mode="l1_loss"),
                 encoder_in_channels=3,
                 unmask_weight=0,
                 fft_weight=0,
                 fft_reweight=False,
                 fft_focal=False,
                 fft_unmask_replace=None,
                 fft_unmask_weight=0,
                 init_cfg=None,
                 **kwargs):
        super(A2MIMHead, self).__init__(init_cfg)
        self.encoder_in_channels = encoder_in_channels
        self.unmask_weight = unmask_weight
        self.fft_weight = fft_weight
        self.fft_reweight = fft_reweight
        self.fft_focal = fft_focal
        self.fft_unmask_weight = fft_unmask_weight
        self.fft_unmask_replace = fft_unmask_replace
        assert fft_unmask_replace in [None, 'target', 'prediction', 'mean', 'mixed',]
        assert 0 <= unmask_weight <= 1 and 0 <= fft_unmask_weight <= 1
        if self.unmask_weight < 1:
            if fft_unmask_replace is None and fft_weight > 0:
                self.fft_unmask_replace = 'target'
                print_log("When using the fft loss, `fft_unmask_replace` should " + \
                    "not be None. Reset as `fft_unmask_replace='target'`.")

        if fft is None and fft_weight > 0:
            raise RuntimeError(
                'Failed to import torch.fft. Please install "torch>=1.7.0".')

        # spatial loss
        assert loss is None or isinstance(loss, dict)
        if loss is None:
            loss = dict(
                type='RegressionLoss', loss_weight=1.0, mode="l1_loss")
        self.criterion = build_loss(loss)
        # fft loss
        if fft_focal:
            fft_loss = dict(
                type='FocalFrequencyLoss', loss_weight=1.0, alpha=1.0,
                ave_spectrum=True, log_matrix=True, batch_matrix=True)
        else:
            fft_loss = loss
            if loss["mode"] not in ["l1_loss", "mse_loss", "focal_l1_loss", "focal_mse_loss",]:
                fft_loss['mode'] = "l1_loss"
        self.fft_loss = build_loss(fft_loss)

    def forward(self, x, x_rec, mask):
        # upsampling mask
        scale_h, scale_w = x.size(2) / mask.size(1), x.size(3) / mask.size(2)
        if scale_h > 1:
            mask = mask.repeat_interleave(int(scale_h), 1).repeat_interleave(
                int(scale_w), 2).unsqueeze(1).contiguous()
        else:
            mask = F.interpolate(mask.type_as(x).unsqueeze(1),
                                 scale_factor=(scale_h, scale_w), mode="nearest")
        
        # spatial loss
        if self.unmask_weight > 0.:
            # reweight unmasked patches
            mask_s = mask.clone()
            mask_s = mask_s + (1. - mask_s) * self.unmask_weight
        else:
            mask_s = mask
        loss_rec = self.criterion(x_rec, target=x, reduction_override='none')
        loss_rec = (loss_rec * mask_s).sum() / (mask_s.sum() + 1e-5) / self.encoder_in_channels
        
        # fourier domain loss
        if self.fft_weight > 0:
            # replace unmask patches (with detach)
            x_replace = None
            if self.fft_unmask_replace is not None:
                if self.fft_unmask_replace == 'target':
                    x_replace = x.clone()
                elif self.fft_unmask_replace == 'prediction':
                    x_replace = x_rec.clone().detach()
                elif self.fft_unmask_replace == 'mean':
                    x_replace = x.mean(dim=[2, 3], keepdim=True).expand(x.size())
                elif self.fft_unmask_replace == 'mixed':
                    x_replace = 0.5 * x_rec.clone().detach() + 0.5 * x.clone()
            if self.fft_unmask_weight < 1:
                mask_f = mask.clone()
                mask_f = mask_f + (1. - mask_f) * self.fft_unmask_weight
                x_rec = (x_rec * mask_f) + (x_replace * (1. - mask_f))  # replace unmask tokens
            
            # apply fft loss
            if self.fft_focal:
                loss_fft = self.fft_loss(x_rec, x)
            else:
                f_x = fft.fftn(x, dim=(2, 3), norm='ortho')
                f_x_rec = fft.fftn(x_rec, dim=(2, 3), norm='ortho')
                if self.fft_reweight:
                    loss_fft = self.fft_loss(f_x_rec, target=f_x, reduction_override='none')
                    fft_weight = loss_fft.clone().detach()
                    loss_fft = (fft_weight * loss_fft).mean()
                else:
                    loss_fft = self.fft_loss(f_x_rec, target=f_x, reduction_override='mean')
            loss_rec += self.fft_weight * loss_fft
        
        losses = dict()
        losses['loss'] = loss_rec
        
        return losses
