import numpy as np
import torch

from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def smoothmix(img,
              gt_label,
              alpha=1.0,
              lam=None,
              dist_mode=False,
              **kwargs):
    r""" SmoothMix augmentation.

    "SmoothMix: a Simple Yet Effective Data Augmentation to Train Robust
    Classifiers". In CVPRW, 2020.
    
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

    def gaussian_kernel(kernel_size, rand_w, rand_h, sigma):
        s = kernel_size * 2
        x_cord = torch.arange(s)
        x_grid = x_cord.repeat(s).view(s, s)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).cuda()
        xy_grid = torch.roll(xy_grid, rand_w, 0)
        xy_grid = torch.roll(xy_grid, rand_h, 1)
        crop_size = s // 4
        xy_grid = xy_grid[crop_size: s - crop_size, crop_size: s - crop_size]

        mean = (s - 1) / 2
        var = sigma ** 2
        g_filter = torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * var))
        g_filter = g_filter.view(kernel_size, kernel_size)
        
        return g_filter

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
        _, _, h, w = img.size()
        y_a = gt_label
        y_b = gt_label[rand_index]
        
        rand_w = int(torch.randint(0, w, (1,)) - w / 2)
        rand_h = int(torch.randint(0, h, (1,)) - h / 2)
        sigma = ((torch.rand(1) / 4 + 0.25) * h).cuda()
        kernel = gaussian_kernel(h, rand_h, rand_w, sigma).cuda()
        img = img * (1 - kernel) + img_ * kernel
        lam = torch.sum(kernel) / (h * w)

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
        _, _, h, w = img.size()
        rand_w = int(torch.randint(0, w, (1,)) - w / 2)
        rand_h = int(torch.randint(0, h, (1,)) - h / 2)
        sigma = (torch.rand(1) / 4 + 0.25) * h
        kernel = gaussian_kernel(h, rand_h, rand_w, sigma).cuda()
        img = img * (1 - kernel) + img_ * kernel
        lam = torch.sum(kernel) / (h * w)
        
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)
