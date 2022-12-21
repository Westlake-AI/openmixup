import numpy as np
import random
import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with `P_1` locations
    `x\in\mathbb{R}^{D_1}` and `P_2` locations `y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT plan.

    Modified from `https://github.com/gpeyre/SinkhornAutoDiff`

    Args:
        eps (float): regularization coefficient.
        max_iter (int): maximum number of Sinkhorn iterations.
        reduction (string, optional): Specifies the reduction to apply to
            the output: 'none' | 'mean' | 'sum'. 'none': no reduction will
            be applied, 'mean': the sum of the output will be divided by the
            number of elements in the output, 'sum': the output will be summed.
            Defaults to 'none'.
    Shape:
        - Input: `(N, P_1, D_1)`, `(N, P_2, D_2)`.
        - Output: `(N)` or `()`, depending on `reduction`.
    """
    def __init__(self, eps=0.1, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(
                self._M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(
                self._M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self._M(C, U, V))

        return pi

    def _M(self, C, u, v):
        # "Modified cost for logarithmic updates"
        # "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2).cuda()
        y_lin = y.unsqueeze(-3).cuda()
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def mixup_aligned(feat1, feat2, lam, eps=0.1, max_iter=100):
    """AlignMix algorithm"""
    B, C, H, W = feat1.shape
    feat1 = feat1.view(B, C, -1)  # b x C x HW
    feat2 = feat2.view(B, C, -1)  # b x C x HW

    sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter, reduction=None)
    P = sinkhorn(  # optimal plan b x HW x HW
        feat1.permute(0, 2, 1), feat2.permute(0, 2, 1)).detach()
    P = P * (H * W)  # assignment matrix

    # uniformly choose at random, which alignmix to perform
    align_mix = random.randint(0, 1)
    if (align_mix == 0):
        # \tilde{A} = A'R^{T}
        f1 = torch.matmul(feat2, P.permute(0, 2, 1).cuda()).view(B, C, H, W) 
        final = feat1.view(B, C, H, W) * lam + f1 * (1 - lam)
    elif (align_mix == 1):
        # \tilde{A}' = AR
        f2 = torch.matmul(feat1, P.cuda()).view(B, C, H, W).cuda()
        final = f2 * lam + feat2.view(B, C, H, W) * (1 - lam)

    return final


def alignmix(img,
             gt_label,
             alpha=1.0,
             lam=None,
             dist_mode=False,
             eps=0.1,
             max_iter=100,
             **kwargs):
    r""" AlignMix augmentation

    "AlignMixup: Improving Representations By Interpolating Aligned Features
    (http://arxiv.org/abs/2103.15375)". In CVPR, 2022.
        https://github.com/shashankvkt/AlignMixup_CVPR22

    Args:
        img (Tensor): Input images of shape (N, C, H, W). In AlignMix, `img`
            denotes feature maps to perform alignment (instread of ManifoldMix).
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        eps (float): Regularization coefficient for SinkhornDistance.
        max_iter (int): Maximum number of Sinkhorn iterations.
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
        
        feat = mixup_aligned(img, img_, lam, eps, max_iter)
        return feat, (y_a, y_b, lam)
    else:
        raise ValueError("AlignMix cannot perform distributed mixup.")
