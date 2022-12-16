# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weight_reduce_loss, convert_to_one_hot


def asymmetric_loss(pred,
                    target,
                    weight=None,
                    gamma_pos=1.0,
                    gamma_neg=4.0,
                    clip=0.05,
                    disable_grad_focal=True,
                    reduction='mean',
                    avg_factor=None):
    r"""asymmetric loss.

    Please refer to the `paper <https://arxiv.org/abs/2009.14119>`__ for
    details.

    Args:
        pred (torch.Tensor): The predicted logits with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*), (multi-label binarized vector).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Dafaults to None.
        gamma_pos (float): positive focusing parameter. Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We usually set
            gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        disable_grad_focal (bool): Whether to disable grad when caculate the
            gradient-related weights for ACL.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    eps = 1e-8
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    if clip and clip > 0:
        pt = (1 - pred_sigmoid +
              clip).clamp(max=1) * (1 - target) + pred_sigmoid * target
    else:
        pt = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    asymmetric_weight = (1 - pt).pow(gamma_pos * target + gamma_neg *
                                     (1 - target))
    loss = -torch.log(pt.clamp(min=eps)) * asymmetric_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class AsymmetricLoss(nn.Module):
    """asymmetric loss.

    Asymmetric Loss For Multi-Label Classification. In ICCV, 2021.
    <https://arxiv.org/abs/2009.14119>

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss into
            a scalar.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 gamma_pos=0.0,
                 gamma_neg=4.0,
                 clip=0.05,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.post_process = "sigmoid"  # multi-label classification

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """asymmetric loss."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # should be onehot targets
        num_classes = pred.size(-1)
        target = convert_to_one_hot(target, num_classes)
        
        loss_cls = self.loss_weight * asymmetric_loss(
            pred,
            target,
            weight,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            clip=self.clip,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls


@LOSSES.register_module()
class ASLSingleLabel(nn.Module):
    """ ASL exteneded version single-label classification problems
    *** using 'exp' instead of 'sigmoid' ***

    Asymmetric Loss For Multi-Label Classification. In ICCV, 2021.
    <https://arxiv.org/abs/2009.14119>

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        label_smooth_val (float): The degree of label smoothing.
        disable_grad_focal (bool): Whether to disable grad when caculate
            gradient-related weights for ACL.
        reduction (str): The method used to reduce the loss into
            a scalar.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """
    def __init__(self,
                 gamma_pos=0.0,
                 gamma_neg=4.0,
                 clip=None,
                 label_smooth_val=0,
                 disable_grad_focal=True,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(ASLSingleLabel, self).__init__()

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.label_smooth_val = label_smooth_val
        self.disable_grad_focal = disable_grad_focal
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.post_process = "softmax"

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """
            pred (tensor): The predicted logits of [N, C].
            target (tensor): The onehot labels of [N, C] (binarized vector).
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        # should be onehot targets
        num_classes = pred.size(-1)
        target = convert_to_one_hot(target, num_classes)

        # Calculating Probabilities
        log_preds = self.logsoftmax(pred)
        anti_target = 1 - target
        xs_pos = torch.exp(log_preds)  # using 'exp' instead of 'sigmoid'
        xs_neg = 1 - xs_pos

        # no Asymmetric Clipping for single labels

        # ASL weights
        if self.disable_grad_focal:
            torch.set_grad_enabled(False)
        xs_pos = xs_pos * target
        xs_neg = xs_neg * anti_target
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                self.gamma_pos * target + self.gamma_neg * anti_target)
        if self.disable_grad_focal:
            torch.set_grad_enabled(True)
        log_preds = log_preds * asymmetric_w
        
        # label smoothing
        if self.label_smooth_val > 0:
            target = target.mul(1 - self.label_smooth_val).add(
                                self.label_smooth_val / num_classes)

        # loss calculation
        loss = - target.mul(log_preds)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
