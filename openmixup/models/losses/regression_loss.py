import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


def l1_loss(pred, target, weight=None, reduction='mean', **kwargs):
    """Calculate L1 loss."""
    loss = F.l1_loss(pred, target, reduction='none')
    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def mse_loss(pred, target, weight=None, reduction='mean', **kwargs):
    """Calculate MSE (L2) loss."""
    loss = F.mse_loss(pred, target, reduction='none')
    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def focal_l1_loss(pred, target,
                  alpha=0.2, gamma=1.0, activate='sigmoid', residual=False,
                  weight=None, reduction='mean', **kwargs):
    r"""Calculate Focal L1 loss.

    Delving into Deep Imbalanced Regression. In ICML, 2021.
    <https://arxiv.org/abs/2102.09554>

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        alpha (float): A balanced form for Focal Loss. Defaults to 0.2.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 1.0.
        activate (str): activate methods in Focal loss in {'sigmoid', 'tanh'}.
            Defaults to 'sigmoid'.
        residual (bool): Whether to use the original l1_loss, i.e., l1 + focal_l1.
            Defaults to False.
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    _loss = F.l1_loss(pred, target, reduction='none')
    if activate == 'tanh':
        loss = _loss * (torch.tanh(alpha * _loss)) ** gamma
    else:
        loss = _loss * (2. * torch.sigmoid(alpha * _loss) - 1.) ** gamma
    if residual:
        loss += _loss

    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def focal_mse_loss(pred, target,
                   alpha=0.2, gamma=1.0, activate='sigmoid', residual=False,
                   weight=None, reduction='mean', **kwargs):
    r"""Calculate Focal MSE (L2) loss.

    Delving into Deep Imbalanced Regression. In ICML, 2021.
    <https://arxiv.org/abs/2102.09554>

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        alpha (float): A balanced form for Focal Loss. Defaults to 0.2.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 1.0.
        activate (str): activate methods in Focal loss in {'sigmoid', 'tanh'}.
            Defaults to 'sigmoid'.
        residual (bool): Whether to use the original l2_loss, i.e., l2 + focal_l2.
            Defaults to False.
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    _l2_loss = (pred - target) ** 2
    _l1_loss = torch.abs(pred - target)
    if activate == 'tanh':
        loss = _l2_loss * (torch.tanh(alpha * _l1_loss)) ** gamma
    else:
        loss = _l2_loss * (2. * torch.sigmoid(alpha * _l1_loss) - 1.) ** gamma
    if residual:
        loss += _l2_loss
    
    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def huber_loss(pred,
               target,
               beta=1.0,
               weight=None,
               reduction='mean',
               **kwargs):
    r"""Calculate Huber loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        beta (float): Weight factor of the L1 and L2 losses.
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    _loss = torch.abs(pred - target)
    cond = _loss < beta
    loss = torch.where(cond, 0.5 * _loss ** 2 / beta, _loss - 0.5 * beta)

    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def balanced_l1_loss(pred, target,
                     alpha=0.5, beta=1.0, gamma=1.5,
                     weight=None, reduction='mean', **kwargs):
    r"""Calculate Balanced L1 loss.

    Libra R-CNN: Towards Balanced Learning for Object Detection. In CVPR, 2019.
    <https://arxiv.org/abs/1904.02701>

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        beta (float): The loss is a piecewise function of prediction and
            target and ``beta`` serves as a threshold for the difference
            between the prediction and target. Defaults to 1.0.
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss.
            Defaults to 1.5.
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert beta > 0 and alpha > 0
    if target.numel() == 0:
        return pred.sum() * 0

    _loss = torch.abs(pred - target)
    b = math.e ** (gamma / alpha) - 1
    loss = torch.where(
        _loss < beta, alpha / b * (b * _loss + 1) * torch.log(b * _loss / beta + 1) - \
            alpha * _loss, gamma * _loss + gamma / b - alpha * beta)
    
    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def kl_loss(pred,
            target,
            weight=None,
            reduction='batchmean',
            **kwargs):
    r"""Calculate KL loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    loss = F.kl_div(pred, target, reduction='none', log_target=False)

    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'batchmean':
        loss = loss.sum() / pred.size(0)
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def general_kl_loss(pred,
                    target,
                    alpha=0.1,
                    weight=None,
                    reduction='mean',
                    **kwargs):
    r"""Calculate General KL loss.

    GenURL: A General Framework for Unsupervised Representation Learning.
    <https://arxiv.org/abs/2110.14553>

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        alpha (float): Weight factor of the KL and sum losses.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    EPS = 1e-10
    # remove negative target
    if (target < 0.).any() == True:  # min-max normalization
        B, C = target.shape[:2]
        t_min, _ = torch.min(target.view(B, C, -1), dim=2)
        t_max, _ = torch.max(target.view(B, C, -1), dim=2)
        target = (target - t_min.view(B, C, 1)) / \
            (t_max.view(B, C, 1) - t_min.view(B, C, 1))

    # element-wise losses
    sum1 = - (pred * torch.log(target + EPS))
    sum2 = F.l1_loss(pred, target)
    loss = sum1 + alpha * sum2

    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'batchmean':
        loss = loss.sum() / pred.size(0)
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def fuzzy_ce_loss(pred,
                  target,
                  weight=None,
                  reduction='mean',
                  **kwargs):
    r"""Calculate Fuzzy System Cross-entropy (CE) loss.

    UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
    <https://arxiv.org/abs/1802.03426v1>

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    EPS = 1e-10
    # remove negative target
    if (target < 0.).any() == True:  # min-max normalization
        B, C = target.shape[:2]
        t_min, _ = torch.min(target.view(B, C, -1), dim=2)
        t_max, _ = torch.max(target.view(B, C, -1), dim=2)
        target = (target - t_min.view(B, C, 1)) / \
            (t_max.view(B, C, 1) - t_min.view(B, C, 1))
    
    # element-wise losses
    sum1 = (pred * torch.log(target + EPS))
    sum2 = ((1 - pred) * torch.log(1 - target + EPS))
    loss = -1 * (sum1 + sum2)

    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'batchmean':
        loss = loss.sum() / pred.size(0)
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def charbonnier_loss(pred,
                     target,
                     weight=None,
                     eps=1e-8,
                     reduction='mean',
                     **kwargs):
    r"""Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid
        Networks. In TPAMI, 2018.
    <https://arxiv.org/abs/1710.01992v1>

    Args:
        pred (Tensor): Prediction Tensor with shape (N, \*).
        target ([type]): Target Tensor with shape (N, \*).
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.

    Returns:
        torch.Tensor: The calculated loss.
    """
    loss = torch.sqrt((pred - target) ** 2 + eps)

    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


class BMCLoss(nn.Module):
    r"""Balanced MSE loss (Batch-based Monte-Carlo).

    Balanced MSE for Imbalanced Visual Regression. In CVPR 2022.
    <https://arxiv.org/abs/2203.16427>

    Args:
        init_noise_sigma (float): Initial scale of the noise.
    """

    def __init__(self, init_noise_sigma=1.0):
        super(BMCLoss, self).__init__()
        self.noise_sigma = nn.Parameter(
            torch.tensor(init_noise_sigma), requires_grad=True)
        self.post_process = "none"  # regression

    def bmc_loss(self, pred, target, noise_var):
        logits = - 0.5 * (pred - target.T).pow(2) / noise_var
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
        loss = loss * (2 * noise_var).detach()
        return loss

    def forward(self, pred, target, weight=None, reduction='mean', **kwargs):
        """forward BMC loss.

        Args:
            pred (Tensor): Prediction Tensor with shape (N, \*).
            target ([type]): Target Tensor with shape (N, \*).
            weight (tensor): Sample-wise reweight of (N, \*) or element-wise
                reweight of (1, \*). Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        noise_var = self.noise_sigma ** 2
        loss = self.bmc_loss(
            pred.view(pred.size(0), -1), target.view(target.size(0), -1), noise_var)
        
        if weight is not None:
            loss *= weight.expand_as(loss)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss


class BNILoss(nn.Module):
    r"""Balanced MSE loss (Bin-based Numerical Integration).

    Balanced MSE for Imbalanced Visual Regression. In CVPR 2022.
    <https://arxiv.org/abs/2203.16427>

    Args:
        init_noise_sigma (float): Initial scale of the noise.
        bucket_centers (np.array): Pre-defined bin centers.
        bucket_weights (np.array): Pre-defined weight for each bin.
    """

    def __init__(self,
                 init_noise_sigma=1.0,
                 bucket_centers=None, bucket_weights=None):
        super(BNILoss, self).__init__()
        self.noise_sigma = nn.Parameter(
            torch.tensor(init_noise_sigma), requires_grad=True)
        self.bucket_centers = torch.tensor(bucket_centers).cuda()
        self.bucket_weights = torch.tensor(bucket_weights).cuda()
        self.post_process = "none"  # regression

    def bni_loss(self, pred, target, noise_var, bucket_centers, bucket_weights):
        mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var

        num_bucket = bucket_centers.shape[0]
        bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
        bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

        balancing_term = - 0.5 * (
            pred.expand(-1, num_bucket) - bucket_center).pow(2) \
            / noise_var + bucket_weights.log()
        balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
        loss = mse_term + balancing_term
        loss = loss * (2 * noise_var).detach()
        return loss

    def forward(self, pred, target, weight=None, reduction='mean', **kwargs):
        """forward BNI loss.

        Args:
            pred (Tensor): Prediction Tensor with shape (N, \*).
            target ([type]): Target Tensor with shape (N, \*).
            weight (tensor): Sample-wise reweight of (N, \*) or element-wise
                reweight of (1, \*). Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        noise_var = self.noise_sigma ** 2
        loss = self.bni_loss(
            pred.view(pred.size(0), -1), target.view(target.size(0), -1),
            noise_var, self.bucket_centers, self.bucket_weights)
        
        if weight is not None:
            loss *= weight.expand_as(loss)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss


@LOSSES.register_module()
class RegressionLoss(nn.Module):
    r"""Simple Regression Loss.

    Args:
        mode (bool): Type of regression loss. Notice that when using
            FP16 training, {'mse_loss', 'smooth_l1_loss'} should use
            'mmcv' mode. Defaults to "mse_loss".
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 mode="mse_loss",
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(RegressionLoss, self).__init__()
        self.mode = mode
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_kwargs = dict()
        self.norm_loss_list = [
            "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss",
            "charbonnier_loss", "focal_mse_loss", "focal_l1_loss",
            "balanced_l1_loss", "balanced_mse_loss",
        ]
        self.div_loss_list = [
            "kl_loss", "general_kl_loss", "fuzzy_ce_loss",
        ]
        assert mode in self.norm_loss_list + self.div_loss_list

        # loss func
        if self.mode in self.norm_loss_list:
            assert reduction in [None, 'none', 'mean', 'sum']
            if "focal" in self.mode:
                self.loss_kwargs = dict(
                    alpha = kwargs.get('alpha', 0.2),
                    gamma = kwargs.get('gamma', 1.0),
                    activate = kwargs.get('activate', 'sigmoid'),
                    residual = kwargs.get('residual', False),
                )
                self.criterion = eval(self.mode)
            elif self.mode == "charbonnier_loss":
                self.loss_kwargs['eps'] = kwargs.get('eps', 1e-10)
                self.criterion = eval(self.mode)
            elif self.mode == "balanced_l1_loss":
                self.loss_kwargs = dict(
                    beta = kwargs.get('beta', 1.0),
                    alpha = kwargs.get('alpha', 0.5),
                    gamma = kwargs.get('gamma', 1.5),
                )
                self.criterion = eval(self.mode)
            elif self.mode == "balanced_mse_loss":
                if kwargs.get("mode", "BMC") == "BMC":
                    self.criterion = BMCLoss(
                        init_noise_sigma=kwargs.get("init_noise_sigma", 1.)
                    )
                else:
                    self.criterion = BNILoss(
                        init_noise_sigma=kwargs.get("init_noise_sigma", 1.),
                        bucket_centers=kwargs.get("bucket_centers", None),
                        bucket_weights=kwargs.get("bucket_weights", None),
                    )
            else:
                self.criterion = eval(self.mode)
        else:
            assert reduction in [None, 'none', 'mean', 'batchmean', 'sum']
            if self.mode == "general_kl_loss":
                self.loss_kwargs['alpha'] = kwargs.get('alpha', 0.1)
            self.criterion = eval(self.mode)
        self.post_process = "none"  # regression

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                **kwargs):
        """caculate loss
        
        Args:
            pred (tensor): Predicted logits of (N, \*).
            target (tensor): Groundtruth label of (N, \*).
            weight (tensor): Sample-wise reweight of (N, \*) or element-wise reweight
                of (1, \*). Defaults to None.
            reduction_override (str): Reduction methods.
        """
        assert reduction_override in (None, 'none', 'mean', 'batchmean', 'sum',)
        reduction = (
            reduction_override if reduction_override is not None else self.reduction)
        kwargs.update(self.loss_kwargs)

        loss_reg = self.loss_weight * self.criterion(
            pred, target, weight=weight, reduction=reduction, **kwargs)

        return loss_reg
