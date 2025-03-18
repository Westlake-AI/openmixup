import torch
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
import numpy as np
from mmcv.cnn import kaiming_init, normal_init
from mmcv.runner import BaseModule
=======
from mmcv.cnn import kaiming_init, normal_init
from mmcv.runner import BaseModule

>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
from ..utils import (accuracy, accuracy_mixup, accuracy_semantic_softmax,
                     trunc_normal_init)
from ..registry import HEADS
from ..builder import build_loss


class BaseClsHead(BaseModule):
    """The Basic Classification Head.
       *** Mixup and multi-label classification are supported ***

    Args:
        with_avg_pool (bool): Whether to use GAP before this head.
        loss (dict): Config of classification loss.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int or None): Number of hidden channels in a MLP ClsHead.
        num_classes (int): Number of categories excluding the category.
        multi_label (bool): Whether to use one_hot like labels (requiring the
            multi-label classification loss). Notice that we support the
            single-label cls task to use the multi-label cls loss.
        finetune (bool): Whether to use the finetune mode of ViTs.
        aug_test (bool): Whether to perform test time augmentations.
        frozen (bool): Whether to freeze the parameters.
        **kwargs: Support other args for ClsHead variants.
    """

    def __init__(self,
                 with_avg_pool=False,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 in_channels=2048,
                 hidden_dim=None,
                 num_classes=1000,
                 multi_label=False,
                 finetune=False,
                 aug_test=False,
                 frozen=False,
                 init_cfg=None,
                 **kwargs):
        super(BaseClsHead, self).__init__(init_cfg=init_cfg)
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.finetune = finetune
        self.aug_test = aug_test
        self.frozen = frozen

        # loss
        if loss is not None:
            assert isinstance(loss, dict)
            self.criterion = build_loss(loss)
        else:
            assert multi_label == False
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
            self.criterion = build_loss(loss)
        self.processor = getattr(self.criterion, "processor", None)

        # TODO: build a classification head

        # post-process for inference
        post_process = getattr(self.criterion, "post_process", "none")
        if post_process == "softmax":
            self.post_process = nn.Softmax(dim=1)
        else:
            self.post_process = nn.Identity()

    def _freeze(self):
        """ freeze classification heads """
        raise NotImplementedError

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        raise NotImplementedError

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        raise NotImplementedError

    def forward(self, x, post_process=False, **kwargs):
        """Inference.

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
        # test-time augmentation
        if len(x) > 1 and self.aug_test:
            aug_pred = [self.forward_head(_x, post_process) for _x in x]
            aug_pred = torch.stack(aug_pred).mean(dim=0)
            return [aug_pred]
        # simple test
        else:
            return [self.forward_head(x[0], post_process)]

<<<<<<< HEAD
    def loss(self, cls_score, labels, multi_lam=False, **kwargs):
        """" cls loss forward
        
=======
    def loss(self, cls_score, labels, **kwargs):
        """" classification loss forward

>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
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
<<<<<<< HEAD
        
        # computing loss
        if not isinstance(labels, tuple):
=======

        # computing loss
        if self.processor is not None:
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1)
            assert single_label, "the semantic softmax needs single labels"
            # onehot cls with semantic_softmax
            losses['loss'] = self.criterion(cls_score, labels)
            # compute accuracy
            losses['acc'] = accuracy_semantic_softmax(cls_score, labels, self.processor)
        elif not isinstance(labels, tuple):
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
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
            if len(labels) == 3:
                y_a, y_b, lam = labels
<<<<<<< HEAD
            elif len(labels) == 4:  # lam sum no equal 1
                y_a, y_b, lam, lam_ = labels
            if isinstance(lam, torch.Tensor):  # lam is scalar or tensor [N,\*]
                lam = lam.view(-1, 1)
=======
                if isinstance(lam, torch.Tensor):  # lam is scalar or tensor [N,\*]
                    lam = lam.view(-1, 1)
                lam_a, lam_b = lam, 1 - lam
            else:  # len(labels) == 4 and the sum is no equal to 1
                y_a, y_b, lam_a, lam_b = labels
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                y_a.dim() == 1 or (y_a.dim() == 2 and y_a.shape[1] == 1)
            # Notice: we allow the single-label cls using multi-label loss, thus
            # * For single-label or multi-label cls, loss = loss.sum() / N
            avg_factor = y_a.size(0)

<<<<<<< HEAD
            if not self.multi_label and len(labels) == 3:
                losses['loss'] = \
                    self.criterion(cls_score, y_a, avg_factor=avg_factor, **kwargs) * lam + \
                    self.criterion(cls_score, y_b, avg_factor=avg_factor, **kwargs) * (1 - lam)
            elif len(labels) == 4:
                losses['loss'] = torch.mean(
                    self.criterion(cls_score, y_a, avg_factor=avg_factor, **kwargs) * lam + \
                    self.criterion(cls_score, y_b, avg_factor=avg_factor, **kwargs) * lam_
                )
=======
            if not self.multi_label:
                losses['loss'] = \
                    self.criterion(cls_score, y_a, avg_factor=avg_factor, **kwargs) * lam_a + \
                    self.criterion(cls_score, y_b, avg_factor=avg_factor, **kwargs) * lam_b
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
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
<<<<<<< HEAD
            if multi_lam is False:
=======
            if len(labels) == 3:
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
                losses['acc_mix'] = accuracy_mixup(cls_score, labels)
        return losses


@HEADS.register_module
class ClsHead(BaseClsHead):
    """Simplest classifier head, with only one fc layer.
       *** Mixup and multi-label classification are supported ***

    Args:
        with_avg_pool (bool): Whether to use GAP before this head.
        loss (dict): Config of classification loss.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int or None): Number of hidden channels in a MLP ClsHead.
        num_classes (int): Number of categories excluding the category.
        multi_label (bool): Whether to use one_hot like labels (requiring the
            multi-label classification loss). Notice that we support the
            single-label cls task to use the multi-label cls loss.
        finetune (bool): Whether to use the finetune mode of ViTs.
        aug_test (bool): Whether to perform test time augmentations.
        frozen (bool): Whether to freeze the parameters.
    """

    def __init__(self, **kwargs):
        super(ClsHead, self).__init__(**kwargs)

        # build a classification head
        assert self.hidden_dim is None
        if self.num_classes is not None:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        if self.fc is None:
            return
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        if self.finetune:  # finetune for ViTs
            std = 2e-5
            init_linear = 'trunc_normal'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        x = self.fc(x)
        if post_process:
            x = self.post_process(x)
        return x
<<<<<<< HEAD


@HEADS.register_module
class MSVTClsHead(BaseClsHead):
    def __init__(self, **kwargs):
        super(ClsHead, self).__init__(**kwargs)

        # build a classification head
        self.norm1_name, norm1 = build_norm_layer(
            dict(type='LN'), self.in_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)

        assert self.hidden_dim is None
        if self.num_classes is not None:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        if self.fc is None:
            return
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        if self.finetune:  # finetune for ViTs
            std = 2e-5
            init_linear = 'trunc_normal'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        x = x.view(B, -1, self.embed_dim)
        x = torch.mean(x, 1)
        x = self.norm1(x)
        x = self.fc(x)
        if post_process:
            x = self.post_process(x)
        return x
=======
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
