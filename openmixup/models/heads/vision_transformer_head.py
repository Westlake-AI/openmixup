from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.runner import BaseModule, Sequential

from ..utils import accuracy, accuracy_mixup, lecun_normal_init
from ..registry import HEADS
from ..builder import build_loss


@HEADS.register_module
class VisionTransformerClsHead(BaseModule):
    """Simplest classifier head, with only one fc layer.
       *** Mixup and multi-label classification are supported ***
    
    Args:
        with_avg_pool (bool): Whether to use GAP before this head.
        loss (dict): Config of classification loss.
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the category.
        multi_label (bool): Whether to use one_hot like labels (requiring the
            multi-label classification loss). Notice that we support the
            single-label cls task to use the multi-label cls loss.
        frozen (bool): Whether to freeze the parameters.
    """

    def __init__(self,
                 num_classes=1000,
                 in_channels=384,
                 hidden_dim=None,
                 act_cfg=dict(type='Tanh'),
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_label=False,
                 frozen=False,
                 init_cfg=dict(type='TruncNormal', layer='Linear', std=.02),
                 **kwargs):
        super(VisionTransformerClsHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg
        self.multi_label = multi_label
        self._init_layers()

        # loss
        if loss is not None:
            assert isinstance(loss, dict)
            self.criterion = build_loss(loss)
        else:
            assert multi_label == False
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
            self.criterion = build_loss(loss)
        if frozen:
            self.frozen()

    def _init_layers(self):
        if self.hidden_dim is None:
            layers = [('head', nn.Linear(self.in_channels, self.num_classes))]
        else:
            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = Sequential(OrderedDict(layers))

    def init_weights(self):
        super(VisionTransformerClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            lecun_normal_init(
                self.layers.pre_logits,
                mode='fan_in', distribution='truncated_normal')
    
    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        _, cls_token = x
        if self.hidden_dim is None:
            return cls_token
        else:
            x = self.layers.pre_logits(cls_token)
            return self.layers.act(x)
    
    def frozen(self):
        self.layers.eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        x = self.pre_logits(x)
        return [self.layers.head(x)]

    def loss(self, cls_score, labels, **kwargs):
        """" cls loss forward
        
        Args:
            cls_score (list): Score should be [tensor].
            labels (tuple or tensor): Labels should be tensor [N, \*] by default.
                If labels as tuple, it's used for CE mixup, (gt_a, gt_b, lambda).
        """
        single_label = False
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        
        # computing loss
        if not isinstance(labels, tuple):
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1)
            # Notice: we allow the single-label cls using multi-label loss, thus
            # * For single-label cls, loss = loss.sum() / N
            # * For multi-label cls, loss = loss.sum() or loss.mean()
            avg_factor = labels.size(0) if single_label else None

            target = labels.clone()
            if self.multi_label:
                # convert to onehot labels
                if single_label:
                    target = F.one_hot(target, num_classes=self.num_classes)
            # default onehot cls
            losses['loss'] = self.criterion(
                cls_score[0], target, avg_factor=avg_factor, **kwargs)
            # compute accuracy
            losses['acc'] = accuracy(cls_score[0], labels)
        else:
            # mixup classification
            y_a, y_b, lam = labels
            if isinstance(lam, torch.Tensor):  # lam is scalar or tensor [N,1]
                lam = lam.unsqueeze(-1)
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                y_a.dim() == 1 or (y_a.dim() == 2 and y_a.shape[1] == 1)
            # Notice: we allow the single-label cls using multi-label loss, thus
            # * For single-label cls, loss = loss.sum() / N
            # * For multi-label cls, loss = loss.sum() or loss.mean()
            avg_factor = y_a.size(0) if single_label else None

            if not self.multi_label:
                losses['loss'] = \
                    self.criterion(cls_score[0], y_a, avg_factor=avg_factor, **kwargs) * lam + \
                    self.criterion(cls_score[0], y_b, avg_factor=avg_factor, **kwargs) * (1 - lam)
            else:
                # convert to onehot labels
                if single_label:
                    y_a = F.one_hot(y_a, num_classes=self.num_classes)
                    y_b = F.one_hot(y_b, num_classes=self.num_classes)
                # mixup onehot like labels, using a multi-label loss
                y_mixed = lam * y_a + (1 - lam) * y_b
                losses['loss'] = self.criterion(
                    cls_score[0], y_mixed, avg_factor=avg_factor, **kwargs)
            # compute accuracy
            losses['acc'] = accuracy(cls_score[0], labels[0])
            losses['acc_mix'] = accuracy_mixup(cls_score[0], labels)
        return losses
