import torch
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F


@torch.no_grad()
def snapmix(img,
            gt_label,
            alpha=1.0,
            lam=None,
            dist_mode=False,
            features=None,
            **kwargs):
    r""" SnapMix augmentation

    "SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data"
    Fine-grained Image Classification (https://arxiv.org/abs/2012.04846)". In AAAI, 2021.
        https://github.com/Shaoli-Huang/SnapMix
    
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
        features (tensor): Feature maps for attentive regions.
    """

    def rand_bbox(size, lam,center=False,attcen=None):
        if len(size) == 4:
            W, H = size[2], size[3]
        elif len(size) == 3:
            W, H = size[1], size[2]
        elif len(size) == 2:
            W, H = size[0], size[1]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

        if attcen is None:
            # uniform
            cx, cy = 0, 0
            if W > 0 and H > 0:
                cx, cy = np.random.randint(W), np.random.randint(H)
            if center:
                cx, cy = int(W/2), int(H/2)
        else:
            cx = attcen[0]
            cy = attcen[1]

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    if lam is None:
        lam = np.random.beta(alpha, alpha)
        lam_ = np.random.beta(alpha, alpha)

    b, _, h, w = img.size()
    lam_a = torch.ones(img.size(0))
    lam_b = 1 - lam_a

    rand_index = torch.randperm(b).cuda()

    features_ = features[rand_index, :, :]
    y_a = gt_label
    y_b = gt_label[rand_index].clone()

    same_label = y_a == y_b

    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
    bbx1_, bby1_, bbx2_, bby2_ = rand_bbox(img.size(), lam_)

    if not dist_mode:
        area = (bby2 - bby1) * (bbx2 - bbx1)
        area_ = (bby2_ - bby1_) * (bbx2_ - bbx1_)
        if area_ > 0 and area > 0:
            ncont = img[rand_index, :, bbx1_:bbx2_, bby1_:bby2_].clone()
            ncont = F.interpolate(ncont, size=(bbx2 - bbx1, bby2 - bby1), mode='bilinear', 
                                align_corners=True)
            
            img[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - features[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (features.sum(2).sum(1) + 1e-8)
            lam_b = features_[:, bbx1_:bbx2_, bby1_:bby2_].sum(2).sum(1) / (features_.sum(2).sum(1) + 1e-8)
            tmp = lam_a.clone()

            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1 - lam

        return img, (y_a, y_b, lam_a, lam_b)
