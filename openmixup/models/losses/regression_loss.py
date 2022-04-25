import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


def general_kl_loss(pred,
                    target,
                    alpha=0.1,
                    reduction='mean',
                    **kwargs):
    r"""Calculate General KL loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        alpha (float): Weight factor of the KL and sum losses.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    EPS = 1e-10
    # element-wise losses    
    if (target < 0.).any() == True:  # min-max normalization
        B, C, _, _ = target.size()
        t_min, _ = torch.min(target.view(B, C, -1), dim=2)
        t_max, _ = torch.max(target.view(B, C, -1), dim=2)
        target = (target - t_min.view(B, C, 1, 1)) / \
            (t_max.view(B, C, 1, 1) - t_min.view(B, C, 1, 1))

    sum1 = - (pred * torch.log(target + EPS))
    sum2 = F.l1_loss(pred, target)
    loss = (1 - alpha) * sum1 + alpha * sum2

    if reduction == 'mean':  # 'benchmean'
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss


def fuzzy_ce_loss(pred,
                  target,
                  reduction='mean',
                  **kwargs):
    r"""Calculate Fuzzy System Cross-entropy (CE) loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    EPS = 1e-10
    # element-wise losses    
    if (target < 0.).any() == True:  # min-max normalization
        B, C, _, _ = target.size()
        t_min, _ = torch.min(target.view(B, C, -1), dim=2)
        t_max, _ = torch.max(target.view(B, C, -1), dim=2)
        target = (target - t_min.view(B, C, 1, 1)) / \
            (t_max.view(B, C, 1, 1) - t_min.view(B, C, 1, 1))
    sum1 = - (pred * torch.log(target + EPS))
    sum2 = ((1 - pred) * torch.log(1 - target + EPS))
    loss = -1 * (sum1 + sum2)

    if reduction == 'mean':  # 'benchmean'
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
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
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
        self.norm_loss_list = ["mse_loss", "l1_loss", "smooth_l1_loss"]
        self.div_loss_list = ["kl_loss", "general_kl_loss", "fuzzy_ce_loss",]
        assert mode in self.norm_loss_list + self.div_loss_list
        # loss func
        if self.mode in self.norm_loss_list:
            self.criterion = getattr(F, self.mode)
        else:
            if self.mode == "kl_loss":
                self.criterion = F.kl_div
            else:
                self.criterion = eval(self.mode)
    
    def forward(self,
                pred,
                target,
                reduction_override=None,
                **kwargs):
        r"""caculate loss
        
        Args:
            pred (tensor): Predicted logits of (N, \*).
            target (tensor): Groundtruth label of (N, \*).
        """
        assert reduction_override in (None, 'none', 'mean',)
        reduction = (
            reduction_override if reduction_override is not None else self.reduction)
        loss_reg = self.loss_weight * self.criterion(
            pred, target, reduction=reduction, **kwargs)
        return loss_reg
