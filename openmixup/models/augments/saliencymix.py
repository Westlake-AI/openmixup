import numpy as np
import torch

try:
    from cv2.saliency import StaticSaliencyFineGrained_create
except ImportError:
    StaticSaliencyFineGrained_create = None
from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def saliencymix(img,
                gt_label,
                alpha=1.0,
                lam=None,
                dist_mode=False,
                **kwargs):
    r""" SaliencyMix augmentation.

    "SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better
    Regularization (https://arxiv.org/pdf/2006.01791.pdf)". In ICLR, 2021.
        https://github.com/SaliencyMix/SaliencyMix/blob/main/SaliencyMix_CIFAR/saliencymix.py
    
    Args:
        img (Tensor): Input images of shape (C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
    """
    if StaticSaliencyFineGrained_create is None:
        raise RuntimeError(
            'Failed to import `StaticSaliencyFineGrained_create` from cv2 for SaliencyMix. '
            'Please install it by "pip install opencv-contrib-python".')

    def saliency_bbox(img, lam):
        """ generate saliency box by lam """
        size = img.size()
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        # force fp32 when convert to numpy
        img = img.type(torch.float32)

        # initialize OpenCV's static fine grained saliency detector and
        # compute the saliency map
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        maximum_indices = np.unravel_index(
            np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        x = maximum_indices[0]
        y = maximum_indices[1]

        bbx1 = np.clip(x - cut_w // 2, 0, W)
        bby1 = np.clip(y - cut_h // 2, 0, H)
        bbx2 = np.clip(x + cut_w // 2, 0, W)
        bby2 = np.clip(y + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
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
        
        # detect saliency box
        bbx1, bby1, bbx2, bby2 = saliency_bbox(img[rand_index[0]], lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
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
        # detect saliency box
        bbx1, bby1, bbx2, bby2 = saliency_bbox(img_[0], lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)
