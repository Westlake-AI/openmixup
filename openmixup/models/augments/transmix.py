import numpy as np
import torch
import torch.nn as nn

from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def transmix(img,
             gt_label,
             dist_mode=False,
             alpha=1.0,
             mask=None,
             lam=None,
             attn=None,
             patch_shape=None,
             return_mask=False,
             ratio=0.5,
             **kwargs):
    r""" TransMix augmentation.

    "TransMix: Attend to Mix for Vision Transformers
    (https://arxiv.org/abs/2111.09833)". In CVPR, 2022.
        https://github.com/Beckschen/TransMix

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        mask (Tensor): The cuting-based mixup mask of shape (\*, 1, H, W).
            Notice that TransMix only modify mixed labels according to the
            given `mask` and `attn`, which should not be None.
        attn (Tensor): The attention map to adjust mixed labels, which should
            not be None.
        patch_shape (tuple): The patch resolution of the attn map.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
        ratio (float): Reweight ratio of lam0 and lam1. Defaults to 0.5.
    """

    def rand_bbox(size, lam, return_mask=False):
        """ generate random box by lam as CutMix """
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        if not return_mask:
            return bbx1, bby1, bbx2, bby2
        else:
            mask = torch.zeros((1, 1, W, H)).cuda()
            mask[:, :, bbx1:bbx2, bby1:bby2] = 1
            mask = mask.expand(size[0], 1, W, H)  # (N, 1, H, W)
            return bbx1, bby1, bbx2, bby2, mask

    if lam is None and mask is None:
        lam0 = np.random.beta(alpha, alpha)

    # normal mixup process
    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:  # [N, C, H, W]
            img_ = img[rand_index]
        else:
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # * notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        b, _, h, w = img.size()
        y_a = gt_label
        y_b = gt_label[rand_index]

        # CutMix
        if mask is None:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox(img.size(), lam0, True)
            img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
            lam0 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        else:
            img = (1-mask) * img + mask * img_
            lam0 = torch.mean(mask[0, 0, ...]) / (h * w) if lam is None else lam
        if return_mask:
            img = (img, mask)

        # TransMix
        lam1 = lam0
        if attn is not None:
            mask_ = nn.Upsample(size=patch_shape)(mask).view(b, -1).int()
            attn_ = torch.mean(attn[:, :, 0, 1:], dim=1)  # attn from cls_token to images
            # w1, w2 = torch.sum((1-mask_) * attn_, dim=1), torch.sum(mask_ * attn_, dim=1)
            w1, w2 = torch.sum(mask_ * attn_, dim=1), torch.sum((1-mask_) * attn_, dim=1)
            lam1 = w2 / (w1+w2)  # (b,)
        lam = lam0 * ratio + lam1 * (1-ratio)  # we apply y_mix = lam * y_a + (1-lam) * y_b

        return img, (y_a, y_b, lam)

    # dist mixup with cross gpus shuffle
    else:
        if len(img.size()) == 5:  # self-supervised img [N, 2, C, H, W]
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(  # N
                img_, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        else:
            assert len(img.size()) == 4  # normal img [N, C, H, w]
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(  # N
                img, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        b, _, h, w = img.size()

        # CutMix
        if mask is None:
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam0)
            img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
            lam0 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        else:
            img = (1-mask) * img + mask * img_
            lam0 = torch.mean(mask[0, 0, ...]) / (h * w) if lam is None else lam
        if return_mask:
            img = (img, mask)

        # TransMix
        lam1 = lam0
        if attn is not None:
            mask_ = nn.Upsample(size=patch_shape)(mask).view(b, -1).int()
            attn_ = torch.mean(attn[:, :, 0, 1:], dim=1)  # attn from cls_token to images
            w1, w2 = torch.sum((1-mask_) * attn_, dim=1), torch.sum(mask_ * attn_, dim=1)
            lam1 = w2 / (w1+w2)  # (b,)
        lam = lam0 * ratio + lam1 * (1-ratio)

        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)
