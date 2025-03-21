import numpy as np
import torch


@torch.no_grad()
def tla(img,
        gt_label,
        alpha=1.0,
        dist_mode=False,
        patch_size=16,
        lam=None,
        return_mask=False,
        **kwargs):
    r""" Token Labeling Align augmentation.

    "Token-Label Alignment for Vision Transformers"
    Basic Image Classification (https://arxiv.org/abs/2210.06455)". In ICCV, 2023.
        https://github.com/Euphoria16/TL-Align

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
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """
    def rand_token_bbox(img_shape, lam, margin=0., count=None):
        """ Modified CutMix bounding-box(token-level)
        Generates a random square bbox based on lambda value. This impl includes
        support for enforcing a border margin as percent of bbox dimensions.

        Args:
            img_shape (tuple): Image shape as tuple
            lam (float): Cutmix lambda value
            margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
            count (int): Number of bbox to generate
        """
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape[-2:]
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yu = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xu = np.clip(cx + cut_w // 2, 0, img_w)
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
        return (yl, yu, xl, xu), lam

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

        b, c, h, w = img.size()
        y_a = gt_label
        y_b = gt_label[rand_index]

        img = img.view(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
        img = img.permute(0, 1, 2, 4, 3, 5).contiguous()  # .view(B, C, -1, self.patch_size, self.patch_size)
        (yl, yh, xl, xh), lam = rand_token_bbox(img.shape[2:4], lam)
        img[:, :, yl:yh, xl:xh, :, :] = img[rand_index][:, :, yl:yh, xl:xh, :, :]
        img = img.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, -1, h, w)

        mask = torch.zeros([b, 1, h // patch_size, w // patch_size]).cuda()
        mask[:, :, yl:yh, xl:xh] = 1
        mask = torch.nn.functional.interpolate(mask, scale_factor=patch_size, mode='nearest')

        if mask is None:
            img = lam * img + (1 - lam) * img[rand_index]
        else:
            img = (1 - mask) * img + mask * img_

        if return_mask:
            img = (img, mask)
        return img, (y_a, y_b, lam)
