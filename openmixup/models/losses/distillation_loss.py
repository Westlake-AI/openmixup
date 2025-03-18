# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved. 
"""
Implements the knowledge distillation loss
"""
import torch
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F
import math

from ..builder import build_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self,
                 loss=dict(type='LabelSmoothLoss',
                           label_smooth_val=0.1, num_classes=100, mode='original', loss_weight=1.0),
                 distillation_type='none', 
                 alpha=0.5, 
                 tau=1.0):
        super().__init__()
        if loss is not None:
            assert isinstance(loss, dict)
            self.criterion = build_loss(loss)
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    @force_fp32(apply_to=('pred', ))
    def forward(self,
                teacher_outputs,
                outputs,
                outputs_kd,
                labels,
                avg_factor=None,
                reduction='mean',
                **kwargs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        if outputs_kd is None:
            raise ValueError("Know Distillation Outputs is None")

        base_loss = self.criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        if teacher_outputs is None and self.distillation_type is not 'none':
            raise ValueError("Please make sure the teacher model outputs are Tensor")

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        loss = weight_reduce_loss(loss, weight=1.0, reduction=reduction, avg_factor=avg_factor)

        return loss