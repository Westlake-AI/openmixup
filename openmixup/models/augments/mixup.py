import numpy as np
import torch

from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def mixup(img,
          gt_label,
          alpha=1.0,
          lam=None,
          dist_mode=False,
          **kwargs):
    r""" MixUp augmentation.

    "Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)".
    In ICLR, 2018.
        https://github.com/facebookresearch/mixup-cifar10
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
    """
    if lam is None:
        lam = np.random.beta(alpha, alpha)

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

        y_a = gt_label
        y_b = gt_label[rand_index]
        img = lam * img + (1 - lam) * img_
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
        img = lam * img + (1 - lam) * img_
        
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)
