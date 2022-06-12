import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss, convert_to_one_hot


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""Sigmoid focal loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Dafaults to None.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (float): A balanced form for Focal Loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' ,
            loss is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):
    r"""Focal loss.

    Focal Loss for Dense Object Detection. In ICCV 2017.
    <https://arxiv.org/abs/1708.02002>

    Args:
        gamma (float): Focusing parameter in focal loss.
            Defaults to 2.0.
        alpha (float): The parameter in balanced form of focal
            loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none" and "mean". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        r"""Sigmoid focal loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Dafaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # should be onehot targets
        num_classes = pred.size(-1)
        target = convert_to_one_hot(target, num_classes)

        loss_cls = self.loss_weight * sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls


@LOSSES.register_module()
class FocalFrequencyLoss(nn.Module):
    r"""Implements of focal frequency loss

    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for
            flexibility. Default: 1.0
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm.
            Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using
            batch-based statistics. Default: False
    """

    def __init__(self,
                 loss_weight=1.0,
                 alpha=1.0,
                 ave_spectrum=False,
                 log_matrix=False,
                 batch_matrix=False):
        
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def loss_formulation(self, f_pred, f_targ, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            weight_matrix = matrix.detach()  # predefined
        else:
            # if the matrix is calculated online: continuous, dynamic,
            #   based on current Euclidean distance
            matrix_tmp = (f_pred - f_targ) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)
            
            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = \
                    matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            
            matrix_tmp[torch.isnan(matrix_tmp)] = 0.
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()
        
        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))
        
        # frequency distance using (squared) Euclidean distance
        tmp = (f_pred - f_targ) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance

        return loss.mean()

    def forward(self, pred, target, matrix=None, **kwargs):
        r"""Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        f_pred = torch.fft.fft2(pred, dim=(2, 3), norm='ortho')
        f_targ = torch.fft.fft2(target, dim=(2, 3), norm='ortho')
        f_pred = torch.stack([f_pred.real, f_pred.imag], -1)
        f_targ = torch.stack([f_targ.real, f_targ.imag], -1)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            f_pred = torch.mean(f_pred, 0, keepdim=True)
            f_targ = torch.mean(f_targ, 0, keepdim=True)
        
        loss = self.loss_formulation(f_pred, f_targ, matrix) * self.loss_weight

        return loss
