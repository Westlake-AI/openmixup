import numpy as np
import torch
from openmixup.models.utils.augmentation import (autocontrast, equalize, posterize, rotate, solarize,
                                                shear_x, shear_y, translate_x, translate_y, color,
                                                contrast, brightness, sharpness)
from PIL import Image
from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def augmix(img,
           gt_label,
           alpha=1.0,
           lam=None,
           mixture_depth=-1,
           mixture_width=3,
           severity=1,
           dist_mode=False,
           **kwargs):
    r""" AugMix augmentation.

    "AugMix: A Simple Data Processing Method to Improve Robustness and 
    Uncertainty. (https://arxiv.org/abs/1912.02781)". In ICLR, 2020.
        https://github.com/google-research/augmix
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        severity (int): Severity of underlying augmentation operators (between 1 to 10).
        width (int): Width of augmentation chain
        depth (int): Depth of augmentation chain. -1 enables stochastic depth uniformly from [1, 3]
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
    """


    def normalize(image):
        """Normalize input image channel-wise to zero mean and unit variance."""
        w = image.shape[-1]
        if w > 32:
            # ImageNet-1K or Tiny-ImageNet
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]
        else:
            # CIFAR-100
            MEAN = [0.4914, 0.4822, 0.4465]
            STD = [0.2023, 0.1994, 0.2010]

        mean, std = np.array(MEAN), np.array(STD)
        image = (image - mean[:, None, None]) / std[:, None, None]

        return image


    def apply_op(image, op, severity):
        image = np.clip(image * 255., 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image.transpose(1, 2, 0))  # Convert to PIL.Image
        pil_img = op(pil_img, severity)
        return np.asarray(pil_img).transpose(2, 0, 1) / 255.

    augmentations = [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
        translate_x, translate_y, color, contrast, brightness, sharpness
    ]

    ws = np.float32(np.random.dirichlet([alpha] * mixture_width))
    lam = np.float32(np.random.beta(alpha, alpha))

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

        img = img.cpu().numpy()
        mix = np.zeros_like(img)
        for i in range(mixture_width):
            img_aug = img.copy()
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(augmentations)
                for j in range(img_aug.shape[0]):
                    img_aug[j] = apply_op(img_aug[j], op, severity)
            mix += ws[i] * normalize(img_aug)
        img = (1 - lam) * normalize(img) + lam * mix

        return torch.from_numpy(img).cuda().float(), lam
