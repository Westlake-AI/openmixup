import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def attentivemix(img,
                 gt_label,
                 alpha=1.0,
                 lam=None,
                 dist_mode=False,
                 features=None,
                 grid_scale=32,
                 top_k=6,
                 **kwargs):
    r""" AttentiveMix augmentation

    "Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning
    Based Image Classification (https://arxiv.org/abs/2003.13048)". In ICASSP, 2020.
        https://github.com/xden2331/attentive_cutmix
    
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
        grid_scale (float): The upsampling scale of attentive grids.
        top_k (int): Using top_k attentive regions in feature maps.
    """
    
    # basic mixup args
    bs, _, att_size, _ = features.size()
    att_grid = att_size**2
    if att_size * grid_scale != img.size(2):
        grid_scale = img.size(2) / att_size
    if lam is None:
        lam = np.random.beta(alpha, alpha)
    # Notice: official attentivemix uses fixed lam by top_k, while attentivemix+
    #   in this repo uses lam\in\Beta(a,a) to choose top_k for better preformances.
    if top_k is None:
        top_k = min(max(1, int(att_grid * lam)), att_grid)
    
    if not dist_mode:
        # normal mixup process
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:  # [N, C, H, W]
            img_ = img[rand_index]
        else:
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # Notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        y_a = gt_label
        y_b = gt_label[rand_index]
    else:
        raise ValueError("AttentiveMix cannot perform distributed mixup.")

    # select top_k attentive regions
    features = features.mean(1)
    _, att_idx = features.view(bs, att_grid).topk(top_k)
    att_idx = torch.cat([
        (att_idx // att_size).unsqueeze(1),
        (att_idx  % att_size).unsqueeze(1),], dim=1)
    mask = torch.zeros(bs, 1, att_size, att_size).cuda()
    for i in range(bs):
        mask[i, 0, att_idx[i, 0, :], att_idx[i, 1, :]] = 1
    mask = F.upsample(mask, scale_factor=grid_scale, mode="nearest")
    lam = float(mask[0, 0, ...].mean().cpu().numpy())
    img = mask * img + (1 - mask) * img_

    return img, (y_a, y_b, lam)
