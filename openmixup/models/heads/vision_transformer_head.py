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
        # post-process for inference
        post_process = getattr(self.criterion, "post_process", "none")
        if post_process == "softmax":
            self.post_process = nn.Softmax(dim=1)
        else:
            self.post_process = nn.Identity()

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

    def forward(self, x, post_process=False, **kwargs):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features. Multi-stage inputs are acceptable
                but only the last stage will be used to classify. The shape of every
                item should be ``(num_samples, in_channels)``.
            post_process (bool): Whether to do post processing (e.g., softmax) the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.
        """
        assert isinstance(x, (tuple, list)) and len(x) >= 1
        x = x[0]
        x = self.pre_logits(x)
        x = self.layers.head(x)
        if post_process:
            x = self.post_process(x)
        return [x]

    def loss(self, cls_score, labels, **kwargs):
        """" cls loss forward
        
        Args:
            cls_score (list): Score should be [tensor].
            labels (tuple or tensor): Labels should be tensor [N, \*] by default.
                If labels as tuple, it's used for CE mixup, (gt_a, gt_b, lambda).
        """
        single_label = False
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) >= 1
        if len(cls_score) > 1:
            assert isinstance(labels, torch.Tensor), "Only support one-hot labels."
            labels = labels.reshape(labels.size(0), -1).repeat(len(cls_score), 1).squeeze()
            cls_score = torch.cat(cls_score, dim=0)
        else:
            cls_score = cls_score[0]
        
        # computing loss
        if not isinstance(labels, tuple):
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1)
            # Notice: we allow the single-label cls using multi-label loss, thus
            # * For single-label or multi-label cls, loss = loss.sum() / N
            avg_factor = labels.size(0)

            target = labels.clone()
            if self.multi_label:
                # convert to onehot labels
                if single_label:
                    target = F.one_hot(target, num_classes=self.num_classes)
            # default onehot cls
            losses['loss'] = self.criterion(
                cls_score, target, avg_factor=avg_factor, **kwargs)
            # compute accuracy
            losses['acc'] = accuracy(cls_score, labels)
        else:
            # mixup classification
            y_a, y_b, lam = labels
            if isinstance(lam, torch.Tensor):  # lam is scalar or tensor [N,1]
                lam = lam.unsqueeze(-1)
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                y_a.dim() == 1 or (y_a.dim() == 2 and y_a.shape[1] == 1)
            # Notice: we allow the single-label cls using multi-label loss, thus
            # * For single-label or multi-label cls, loss = loss.sum() / N
            avg_factor = y_a.size(0)

            if not self.multi_label:
                losses['loss'] = \
                    self.criterion(cls_score, y_a, avg_factor=avg_factor, **kwargs) * lam + \
                    self.criterion(cls_score, y_b, avg_factor=avg_factor, **kwargs) * (1 - lam)
            else:
                # convert to onehot labels
                if single_label:
                    y_a = F.one_hot(y_a, num_classes=self.num_classes)
                    y_b = F.one_hot(y_b, num_classes=self.num_classes)
                # mixup onehot like labels, using a multi-label loss
                y_mixed = lam * y_a + (1 - lam) * y_b
                losses['loss'] = self.criterion(
                    cls_score, y_mixed, avg_factor=avg_factor, **kwargs)
            # compute accuracy
            losses['acc'] = accuracy(cls_score, labels[0])
            losses['acc_mix'] = accuracy_mixup(cls_score, labels)
        return losses
