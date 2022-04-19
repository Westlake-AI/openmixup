import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


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
        assert mode in ["mse_loss", "l1_loss", "smooth_l1_loss",]
        # loss func
        self.criterion = getattr(F, self.mode)
    
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
