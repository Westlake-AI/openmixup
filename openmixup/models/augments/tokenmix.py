import numpy as np
import torch
import math
import random


@torch.no_grad()
def tokenmix(img,
             gt_label,
             attn=None,
             alpha=1.0,
             dist_mode=False,
             mask_type='block',  # [block, random]
             minimum_tokens=14,
             lam=None,
             return_mask=False,
             **kwargs):
    r""" TokenMix augmentation.

    "TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers"
    Basic Image Classification (https://arxiv.org/abs/2207.08409)". In ECCV, 2022.
        https://github.com/Sense-X/TokenMix

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        attn (Tensor): Attention Maps.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        mask (Tensor): The cuting-based mixup mask of shape (\*, 1, H, W).
            Notice that TransMix only modify mixed labels according to the
            given `mask` and `attn`, which should not be None.
        mask_type (list): Choose one of the mask type [block, random].
        minimum_tokens (int): Setting the minimum tokens of mixing.
        num_classes (int): Total class of the dataset.
        smoothing (float): Smooth value for the one-hot label.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """

    def generate_mask(lam, mask_token_num_start, min_num_patches=1):
        width, height = 14, 14
        min_aspect = 0.3
        log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))
        mask = np.zeros(shape=(width, height), dtype=np.int32)
        mask_ratio = 1 - lam

        num_masking_patches = min(width * height, int(mask_ratio * width * height) + mask_token_num_start)

        max_num_patches = width * height
        mask_count = 0

        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, max_num_patches)
            delta = 0
            for attempt in range(10):
                target_area = random.uniform(min_num_patches, max_mask_patches)
                aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < width and h < height:
                    top = random.randint(0, height - h)
                    left = random.randint(0, width - w)

                    num_masked = mask[top: top + h, left: left + w].sum()
                    # Overlap
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if mask[i, j] == 0:
                                    mask[i, j] = 1
                                    delta += 1

                    if delta > 0:
                        break
            if delta == 0:
                break
            else:
                mask_count += delta
        mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).cuda()

        return mask


    def generate_mask_random(lam, mask_token_num_start=14):
        width, height = 14, 14
        mask = np.zeros(shape=(width * height), dtype=np.int32)
        mask_ratio = 1 - lam

        num_masking_patches = min(width * height, int(mask_ratio * width * height) + mask_token_num_start)

        mask_idx = np.random.permutation(14 * 14)[:num_masking_patches]
        mask[mask_idx] = 1
        mask = mask.reshape(width, width)

        mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).cuda()

        return mask

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

        b, _, h, w = img.size()
        y_a = gt_label
        y_b = gt_label[rand_index]

        if mask_type == 'block':
            mask = generate_mask(lam, minimum_tokens)
        elif mask_type == 'random':
            mask = generate_mask_random(lam, minimum_tokens)
        else:
            raise ValueError(f"unsupported mask type {mask_type}")
        
        mask_224 = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')

        if mask is None:
            img = lam * img + (1 - lam) * img[rand_index]
        else:
            img = ( 1 - mask_224) * img + mask_224 * img_


        # NOTES! We modified some from the offcial codes, since they choose to use Token Labeling,
        # We choose to use attention score according to the paper.
        if attn is not None:
            attn = torch.mean(attn[:, :, 0, 1:], dim=1).reshape(b, 1, 14, 14)

            score_a = mask * attn
            score_b = (1 - mask) * attn[rand_index]
            score_a = torch.sum(score_a.reshape(b, -1), dim=1)
            score_b = torch.sum(score_b.reshape(b, -1), dim=1)

        if return_mask:
            img = (img, mask)

        return img, (y_a, y_b, score_a, score_b)
