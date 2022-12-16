import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss, convert_to_one_hot


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  **kwargs):
    r"""Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_cross_entropy(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       class_weight=None,
                       avg_factor=None,
                       **kwargs):
    r"""Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_mix_cross_entropy(pred,
                           label,
                           weight=None,
                           reduction='mean',
                           class_weight=None,
                           avg_factor=None,
                           eta_weight=None,
                           eps_smooth=1e-3,
                           **kwargs):
    r"""Calculate the Soft Decoupled Mixup CrossEntropy loss using softmax
        The label can be float mixup label (class-wise sum to 1, k-mixup, k>=2).
       *** Warnning: this mixup and label-smoothing cannot be set simultaneously ***

    Decoupled Mixup for Data-efficient Learning. In arXiv, 2022.
    <https://arxiv.org/abs/2203.10761>

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float (mixup one-hot label).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        eta_weight (list): Reweight the global loss in mixup cls loss as,
            loss = loss_local + eta_weight[i] * loss_global[i]. Default to None.
        eps_smooth (float): If using label smoothing, we assume eps < lam < 1-eps.

    Returns:
        torch.Tensor: The calculated loss
    """
    # *** Assume k-mixup in C classes, k >= 2 and k << C ***
    # step 1: remove labels have less than k-hot (mixed between the
    #    same class will result in the original onehot)
    _eps = max(1e-3, eps_smooth)  # assuming _eps < lam < 1-_eps
    mask_one = (label > _eps).sum(dim=-1)
    mix_num = max(mask_one)
    mask_one = mask_one >= mix_num
    if mask_one.sum() < label.size(0):
        pred_one = pred[mask_one==False, :]
        label_one = label[mask_one==False, :]
        pred = pred[mask_one, :]
        label = label[mask_one, :]
        weight_one = None
        if weight is not None:
            weight_one = weight[mask_one==False, ...].float()
            weight = weight[mask_one, ...].float()
    else:
        if weight is not None:
            weight = weight.float()
        pred_one, label_one, weight_one = None, None, None

    # step 2: select k-mixup for the local and global
    bs, cls_num = label.size()  # N, C
    assert isinstance(eta_weight, list)
    # local: between k classes
    mask_lam_k = label > _eps  # [N, N], top k is true
    lam_k = label[0, label[0, :] > _eps]  # [k,] k-mix relevant classes

    # local: original mixup CE loss between C classes
    loss = -label * F.log_softmax(pred, dim=-1)  # [N, N]
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)  # reduce class

    # global: between lam_i and C-k classes
    if len(set(lam_k.cpu().numpy())) == lam_k.size(0) and lam_k.size(0) > 1:
        # *** trivial solution: lam=0.5, lam=1.0 ***
        assert len(eta_weight) == lam_k.size(0), \
            "eta weight={}, lam_k={}".format(eta_weight, lam_k)
        for i in range(lam_k.size(0)):
            # selected (C-k+1), except lam_k[j], where j!=i (k-1)
            mask_lam_i = (label == lam_k[i]) | ~mask_lam_k  # [N, N]
            pred_lam_i  = pred.reshape([1, bs, -1])[:, mask_lam_i].reshape(
                [-1, cls_num+1-lam_k.size(0)])  # [N, C-k+1]
            label_lam_i = label.reshape([1, bs, -1])[:, mask_lam_i].reshape(
                [-1, cls_num+1-lam_k.size(0)])  # [N, C-k+1]
            # convert to onehot
            label_lam_i = (label_lam_i > 0).type(torch.float)
            # element-wise losses
            loss_global = -label_lam_i * F.log_softmax(pred_lam_i, dim=-1)  # [N, C-1]
            if class_weight is not None:
                loss_global *= class_weight
            # eta reweight
            loss += eta_weight[i] * loss_global.sum(dim=-1)  # reduce class
    # apply weight and do the reduction
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    # step 3: original soft CE loss
    if label_one is not None:
        loss_one = -label_one * F.log_softmax(pred_one, dim=-1)
        if class_weight is not None:
            loss_one *= class_weight
        loss_one = loss_one.sum(dim=-1)  # reduce class
        loss_one = weight_reduce_loss(
            loss_one, weight=weight_one, reduction=reduction, avg_factor=avg_factor)
        loss += loss_one

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         **kwargs):
    r"""Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        label (torch.Tensor): The gt label with shape (N, \*).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape [C] or [N, C], C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.dim() == label.dim()
    # Ensure that the size of class_weight is consistent with pred and label to
    # avoid automatic boracast,
    if class_weight is not None:
        if class_weight.dim() == 1:
            N = pred.size()[0]
            class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(
        pred, label, weight=class_weight, reduction='none')
    
    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        use_mix_decouple (bool): Whether to use decoupled mixup version of
            CrossEntropyLoss with the 'soft' CE implementation. Default to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_soft=False,
                 use_mix_decouple=False,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        self.use_mix_decouple = use_mix_decouple
        assert not (
            self.use_soft and self.use_sigmoid
        ), 'use_sigmoid and use_soft could not be set simultaneously'
        if self.use_mix_decouple:
            assert use_soft, \
                "use_mix_decouple requires 'use_soft' to be true"
        
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.post_process = "softmax"
        # loss func
        if self.use_sigmoid:
            self.criterion = binary_cross_entropy
            self.post_process = "sigmoid"  # multi-label classification
        elif self.use_soft:
            self.criterion = soft_mix_cross_entropy \
                if self.use_mix_decouple else soft_cross_entropy
        else:
            self.criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                eta_weight=None,
                avg_factor=None,
                reduction_override=None,
                class_weight_override=None,
                **kwargs):
        r"""caculate loss
        
        Args:
            cls_score (tensor): Predicted logits of (N, C).
            label (tensor): Groundtruth label of (N, \*).
            weight (tensor): Loss weight for each samples of (N,).
            eta_weight (list): Rescale weight for the global loss when
                'use_mix_decouple'=true, loss = loss_local + eta_weight[i] * \
                loss_global[i]. Default: None.
            avg_factor (int, optional): Average factor that is used to average the loss.
                Defaults to None.
            reduction_override (str, optional): The reduction method used to override
                the original reduction method of the loss. Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override is not None else self.reduction)

        class_weight = \
            class_weight_override if class_weight_override is not None \
                else self.class_weight
        if class_weight is not None:
            if isinstance(class_weight, list):  # [C]
                class_weight = cls_score.new_tensor(class_weight)
            else:
                if class_weight.dim() == 1:  # (C,)
                    assert class_weight.size(0) == cls_score.size(1)
                else:  # (N, C)
                    assert class_weight.shape == cls_score.shape
        # BCE version requires onehot targets
        num_classes = cls_score.size(-1)
        if self.use_sigmoid:
            label = convert_to_one_hot(label, num_classes)
            label = label.float()
        # use_mix_decouple version requires eta weight
        if self.use_mix_decouple:
            assert eta_weight is not None, \
                "use_mix_decouple requires 'eta_weight' to be not None"

        loss_cls = self.loss_weight * self.criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            eta_weight=eta_weight,
            **kwargs)
        return loss_cls
