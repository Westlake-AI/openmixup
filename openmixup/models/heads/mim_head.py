import torch
from mmcv.runner import BaseModule
from torch.nn import functional as F

from ..builder import build_loss
from ..registry import HEADS


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

    Args:
        loss (dict): Config of regression loss.
        encoder_in_channels (int): Number of input channels for encoder.
        unmask_weight (float): Loss weight for unmasked patches.
        abs_output (bool): Whether to constrain prediction to non-negative.
    """

    def __init__(self,
                 loss=dict(
                    type='RegressionLoss', loss_weight=1.0, mode="l1_loss"),
                 encoder_in_channels=3,
                 unmask_weight=0,
                 fft_weight=0,
                 high_pass=False,
                 abs_output=False,
                ):
        super(MIMHead, self).__init__()
        self.encoder_in_channels = encoder_in_channels
        self.unmask_weight = unmask_weight
        self.fft_weight = fft_weight
        self.high_pass = high_pass
        self.abs_output = abs_output

        # loss
        if loss is not None:
            assert isinstance(loss, dict)
            self.criterion = build_loss(loss)
        else:
            loss = dict(type='RegressionLoss', loss_weight=1.0, mode="mse_loss")
            self.criterion = build_loss(loss)

    def forward(self, x, x_rec, mask):
        # reweight unmasked patches
        if self.unmask_weight > 0.:
            mask = mask.type_as(x)
            mask += (1. - mask) * self.unmask_weight
        scale_h, scale_w = x.size(2) / mask.size(1), x.size(3) / mask.size(2)
        if scale_h > 1:
            mask = mask.repeat_interleave(int(scale_h), 1).repeat_interleave(
                int(scale_w), 2).unsqueeze(1).contiguous()
        else:
            mask = F.interpolate(mask.type_as(x).unsqueeze(1),
                                 scale_factor=(scale_h, scale_w), mode="nearest")
        if self.abs_output and (x > 0).any():
            x_rec = torch.abs(x_rec)
        
        # spatial loss
        loss_rec = self.criterion(x_rec, target=x, reduction_override='none')
        loss_rec = (loss_rec * mask).sum() / (mask.sum() +
                                          1e-5) / self.encoder_in_channels
        # fourier domain loss
        if self.fft_weight > 0:
            f_x = torch.fft.fftn(x, dim=(2, 3), norm='ortho')
            f_x_rec = torch.fft.fftn(x_rec, dim=(2, 3), norm='ortho')
            if self.high_pass:
                _, _, H, W = x.size()
                fft_mask = torch.ones((H, W)).type_as(x)
                fft_mask[W//2, H//2  ] = 0
                if H % 2 != 0:
                    fft_mask[W//2, H//2-1] = 0
                    fft_mask[W//2-1, H//2  ] = 0
                    fft_mask[W//2-1, H//2-1] = 0
                f_x = torch.roll(f_x, (H//2, W//2), dims=(2, 3))
                f_x = f_x * fft_mask
                f_x_rec = torch.roll(f_x_rec, (H//2, W//2), dims=(2, 3))
                f_x_rec = f_x_rec * fft_mask
            if self.abs_output:
                f_x = torch.abs(f_x)
                f_x_rec = torch.abs(f_x_rec)
            loss_fft = self.criterion(f_x_rec, target=f_x, reduction_override='mean')
            loss_rec += self.fft_weight * loss_fft
        
        losses = dict()
        losses['loss'] = loss_rec
        
        return losses
