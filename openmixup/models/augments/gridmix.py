import math
import random
import numpy as np
import torch

from openmixup.models.utils import batch_shuffle_ddp, to_2tuple


@torch.no_grad()
def gridmix(img,
            gt_label,
            alpha=1.0,
            lam=None,
            n_holes=20,
            hole_aspect_ratio=1.,
            cut_area_ratio=1.,
            cut_aspect_ratio=1.,
            dist_mode=False,
            **kwargs):
    r""" GridMix augmentation.

    "GridMix: Strong regularization through local context mapping
    (https://www.sciencedirect.com/science/article/pii/S0031320320303976)".
    In Pattern Recognition, 2021.
        https://github.com/IlyaDobrynin/GridMixup

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        n_holes (float): The number of holes in crop by X axis.
        hole_aspect_ratio (float | tuple): The hole aspect ratio.
        cut_area_ratio (float | tuple): The percentage of the crop area in the
            second image on a mixed image.
        cut_aspect_ratio (float | tuple): The crop aspect ratio.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
    """

    def rand_grid(lam, size, cut_area_ratio, cut_aspect_ratio,
                  n_holes, hole_aspect_ratio):
        """ generate random box by the crop ratio and the hole hyperparameters """
        W = size[2]
        H = size[3]
        cut_area = int(H * W * cut_area_ratio)
        cut_w = int(np.sqrt(cut_area / cut_aspect_ratio))
        cut_h = int(cut_w * cut_aspect_ratio)

        # uniform
        cx = np.random.random()
        cy = np.random.random()

        xc1 = int((W - cut_w) * cx)
        yc1 = int((H - cut_h) * cy)
        xc2 = xc1 + cut_w
        yc2 = yc1 + cut_h
        width, height = xc2 - xc1, yc2 - yc1
        assert 1 <= n_holes <= width // 2, \
            "The n_holes must in [1, {}], given {}".format(height//2, n_holes)

        # Get patch width, height and ny
        patch_width = math.ceil(width / n_holes)
        patch_height = int(patch_width * hole_aspect_ratio)
        ny = math.ceil(height / patch_height)

        # Calculate ratio of the hole - percent of hole pixels in the patch
        ratio = np.sqrt(1 - lam)

        # Get hole size
        hole_width = int(patch_width * ratio)
        hole_height = int(patch_height * ratio)

        # min 1 pixel and max patch length - 1
        hole_width = min(max(hole_width, 1), patch_width - 1)
        hole_height = min(max(hole_height, 1), patch_height - 1)

        # Make grid mask
        holes = []
        for i in range(n_holes + 1):
            for j in range(ny + 1):
                x1 = min(patch_width * i, width)
                y1 = min(patch_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))

        mask = torch.zeros((1, 1, W, H)).cuda()
        for x1, y1, x2, y2 in holes:
            mask[0, 0, yc1+y1: yc1+y2, xc1+x1: xc1+x2] = 1.

        return mask

    if lam is None:
        lam = np.random.beta(alpha, alpha)

    n_holes = to_2tuple(n_holes)
    hole_aspect_ratio = to_2tuple(hole_aspect_ratio)
    cut_area_ratio = to_2tuple(cut_area_ratio)
    cut_aspect_ratio = to_2tuple(cut_aspect_ratio)
    # random
    n_holes = random.randint(n_holes[0], n_holes[1])
    hole_aspect_ratio = np.random.uniform(hole_aspect_ratio[0], hole_aspect_ratio[1])
    cut_area_ratio = np.random.uniform(cut_area_ratio[0], cut_area_ratio[1])
    cut_aspect_ratio = np.random.uniform(cut_aspect_ratio[0], cut_aspect_ratio[1])

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

        mask = rand_grid(lam, img.size(), cut_area_ratio, cut_aspect_ratio,
                         n_holes, hole_aspect_ratio)
        img = img * (1 - mask) + img_ * mask
        lam = 1 - (mask[0, 0, ...].sum() / (img.shape[-1] * img.shape[-2]))
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
        mask = rand_grid(lam, img.size(), cut_area_ratio, cut_aspect_ratio,
                         n_holes, hole_aspect_ratio)
        img = img * (1 - mask) + img_ * mask
        lam = 1 - (mask[0, 0, ...].sum() / (img.shape[-1] * img.shape[-2]))

        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)
