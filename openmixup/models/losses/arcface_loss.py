import math

import torch
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F

from .cross_entropy_loss import soft_cross_entropy
from ..registry import LOSSES


@LOSSES.register_module()
class ArcFaceLoss(nn.Module):
    r"""ArcFace Classification loss.

    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    In CVPR, 2019. <https://arxiv.org/abs/1801.07698>

    Args:
        s (float): Temperature in the softmax logit. Defaults to 30.
        m (float): Margin in ArcFace loss. Defaults to 0.5.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
    """

    def __init__(self,
                 s: float = 30.0,
                 m: float = 0.5,
                 reduction='mean',
                 loss_weight: float = 1.0,
                 class_weight=None):
        super().__init__()
        self.s = s
        self.m = m
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)  # (t + m) == pi

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.fp16_enabled = False
        self.post_process = "softmax"

    @force_fp32(apply_to=('pred', ))
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        r"""caculate loss
        
        Args:
            pred (tensor): Predicted logits of (N, C).
            label (tensor): Groundtruth label of (N, \*).
            weight (tensor): Loss weight for each samples of (N,).
            avg_factor (int, optional): Average factor that is used to average the loss.
                Defaults to None.
            reduction_override (str, optional): The reduction method used to override
                the original reduction method of the loss. Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        with torch.cuda.amp.autocast(enabled=False):
            if self.class_weight is not None:
                class_weight = pred.new_tensor(self.class_weight)
            else:
                class_weight = None

            with torch.no_grad():
                ont_hot_target = F.one_hot(target, num_classes=pred.size(-1))

            cos_t = pred.clamp(-1, 1)
            cos_t_m = torch.cos(torch.acos(cos_t) + self.m)
            cos_t_m = torch.where(cos_t > self.threshold, cos_t_m,
                                  cos_t - self.mm)

            logit = ont_hot_target * cos_t_m + (1 - ont_hot_target) * cos_t
            logit = logit * self.s

            loss_cls = self.loss_weight * soft_cross_entropy(
                logit,
                ont_hot_target,
                weight=weight,
                reduction=reduction,
                class_weight=class_weight,
                avg_factor=avg_factor)
        return loss_cls
