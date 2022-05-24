import torch
from mmcv.runner import BaseModule
from torch.nn import functional as F

from ..builder import build_loss
from ..registry import HEADS
from openmixup.utils import print_log


@HEADS.register_module
class MAEPretrainHead(BaseModule):
    """Pre-training head for MAE.

    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self, norm_pix=False, patch_size=16):
        super(MAEPretrainHead, self).__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size

    def patchify(self, imgs):

        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

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


@HEADS.register_module
class SimMIMHead(BaseModule):
    """Pretrain Head for SimMIM.

    Args:
        encoder_in_channels (int): Number of input channels for encoder.
    """

    def __init__(self, encoder_in_channels=3):
        super(SimMIMHead, self).__init__()
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
class MIMHead(BaseModule):
    """Head for MIM training.
        V05.10 update: add `unmask_replace` and fix `fft_unmask`.

    Args:
        loss (dict): Config of regression loss.
        encoder_in_channels (int): Number of input channels for encoder.
        unmask_weight (float): Loss weight for unmasked patches.
        unmask_replace (str): Mode to replace unmask patches (detach) in
            {None, 'target', 'prediction', 'mean',}. Defaults to None.
        fft_weight (float): Loss weight for fft reconstruction loss. Default to 0.
        fft_reweight (bool): Whether to use the fft reweight loss. Default to False.
        fft_focal (bool): Whether to adopt the focal fft loss. Default to False.
        fft_unmask (float): Loss weight to caculate fft loss for unmask tokens.
            Default to 1.
    """

    def __init__(self,
                 loss=dict(
                    type='RegressionLoss', loss_weight=1.0, mode="l1_loss"),
                 encoder_in_channels=3,
                 unmask_weight=0,
                 unmask_replace=None,
                 fft_weight=0,
                 fft_reweight=False,
                 fft_focal=False,
                 fft_unmask=1,
                 **kwargs,
                ):
        super(MIMHead, self).__init__()
        self.encoder_in_channels = encoder_in_channels
        self.unmask_weight = unmask_weight
        self.unmask_replace = unmask_replace
        self.fft_weight = fft_weight
        self.fft_reweight = fft_reweight
        self.fft_focal = fft_focal
        self.fft_unmask = fft_unmask
        assert unmask_replace in [None, 'target', 'prediction', 'mean', 'mixed',]
        assert 0 <= unmask_weight <= 1 and 0 <= fft_unmask <= 1
        if self.unmask_weight < 1:
            if unmask_replace is None and fft_weight > 0:
                self.unmask_replace = 'target'
                print_log("When `unmask_weight<1`, `unmask_replace` should not " + \
                    "be None. Reset as `unmask_replace='target'`.")
        
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
            if loss["mode"] not in ["l1_loss", "mse_loss",]:
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
        loss_rec = self.criterion(x_rec, target=x, reduction_override='none')
        # reweight unmasked patches
        if self.unmask_weight > 0.:
            mask_s = mask.clone()
            mask_s = mask_s + (1. - mask_s) * self.unmask_weight
        else:
            mask_s = mask
        loss_rec = (loss_rec * mask_s).sum() / (mask_s.sum() + 1e-5) / self.encoder_in_channels
        
        # fourier domain loss
        if self.fft_weight > 0:
            # replace unmask patches (with detach)
            x_replace = None
            if self.unmask_replace is not None:
                if self.unmask_replace == 'target':
                    x_replace = x.clone()
                elif self.unmask_replace == 'prediction':
                    x_replace = x_rec.clone().detach()
                elif self.unmask_replace == 'mean':
                    x_replace = x.mean(dim=[2, 3], keepdim=True).expand(x.size())
                elif self.unmask_replace == 'mixed':
                    x_replace = 0.5 * x_rec.clone().detach() + 0.5 * x.clone()
            if self.fft_unmask < 1:
                mask_f = mask.clone()
                mask_f = mask_f + (1. - mask_f) * self.fft_unmask
                x_rec = (x_rec * mask_f) + (x_replace * (1. - mask_f))  # replace unmask tokens
            
            # apply fft loss
            if self.fft_focal:
                loss_fft = self.fft_loss(x_rec, x)
            else:
                f_x = torch.fft.fftn(x, dim=(2, 3), norm='ortho')
                f_x_rec = torch.fft.fftn(x_rec, dim=(2, 3), norm='ortho')
                if self.fft_reweight:
                    loss_fft = self.fft_loss(f_x_rec, target=f_x, reduction_override='none')
                    fft_weight = torch.abs(((f_x - f_x_rec) ** 2).sqrt()).detach()
                    loss_fft = (fft_weight * loss_fft).mean()
                else:
                    loss_fft = self.fft_loss(f_x_rec, target=f_x, reduction_override='mean')
            loss_rec += self.fft_weight * loss_fft
        
        losses = dict()
        losses['loss'] = loss_rec
        
        return losses
