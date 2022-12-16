import torch
import torch.nn as nn
import torch.nn.functional as F
from builtins import NotImplementedError
from mmcv.cnn import kaiming_init, normal_init, trunc_normal_init
from mmcv.runner import BaseModule

from ..utils import accuracy, accuracy_mixup
from ..registry import HEADS
from ..builder import build_loss


@HEADS.register_module
class ClsMixupHead(BaseModule):
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
        two_hot (bool): Whether to use multi-hot label (two hot).
        two_hot_scale (float): Rescale the sum of labels, in (0, 1]. The sum of
            softmax labels is 1, while that of the two-hot labels is 2. This scalar
            is used to rescale the sum of labels to (0, 2].
        lam_scale_mode (str): The mode of rescaling two-hot or soft mixup labels,
            in {'pow', 'exp', 'none'}. If mode!='none', rescaling the labels with
            lam_thr and lam_idx. Default: "none".
        lam_thr (float): Rescale threshold for two-hot labels, in [0,1].
        lam_idx (float): Rescale factor for the exp or power function.
        eta_weight (dict): The lam threhold of whether to use the eta weights. It
            contains 'eta_weight=dict(eta=1, mode="both", thr=1)', where 'eta' denotes
            the basic rescale factor of each lam term and 'mode' is the selection method.
                If eta_weight['mode']=="both", add the eta_weight for the both lam term.
                If eta_weight['mode']=="less", add the eta_weight for lam < thr.
                If eta_weight['mode']=="more", add the eta_weight for lam > thr.
            Default: dict(eta=1, mode="both", thr=0).
        neg_weight (bool or float): Whether to remove (or reweight) the negative
            part of loss according to gt_label (should be BCE multi-label loss).
            Default: 1 (True).
        aug_test (bool): Whether to perform test time augmentations.
        frozen (bool): Whether to freeze the parameters.
    """

    def __init__(self,
                 with_avg_pool=False,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 in_channels=2048,
                 num_classes=1000,
                 multi_label=False,
                 two_hot=False,
                 two_hot_scale=1,
                 lam_scale_mode='none',
                 lam_thr=1,
                 lam_idx=1,
                 eta_weight=dict(eta=0, mode="both", thr=0.5),
                 neg_weight=1,
                 aug_test=False,
                 frozen=False,
                 init_cfg=None):
        super(ClsMixupHead, self).__init__(init_cfg=init_cfg)
        self.with_avg_pool = bool(with_avg_pool)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.multi_label = bool(multi_label)
        self.two_hot = bool(two_hot)
        self.two_hot_scale = float(two_hot_scale)
        self.lam_scale_mode = str(lam_scale_mode)
        self.lam_thr = float(lam_thr)
        self.lam_idx = float(lam_idx)
        self.eta_weight = eta_weight
        self.neg_weight = float(neg_weight) if float(neg_weight) != 1 else 1
        self.aug_test = aug_test
        assert lam_scale_mode in ['none', 'pow', 'exp']
        assert eta_weight["mode"] in ['more', 'less', 'both'] and \
            0 <= eta_weight["thr"] <= 1 and eta_weight["eta"] < 100
        assert 0 < lam_thr <= 1 and -100 < lam_idx < 100
        assert 0 < two_hot_scale <= 1 and 0 <= neg_weight <= 1

        # loss
        if loss is not None:
            assert isinstance(loss, dict)
            self.criterion = build_loss(loss)
        else:
            assert multi_label == False
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
            self.criterion = build_loss(loss)
        if self.neg_weight != 1:
            0 <= self.neg_weight <= 1, "the weight of negative parts should not be \
                larger than the postive part."
            assert multi_label == True and loss['type'] == 'CrossEntropyLoss'
        # fc layer
        self.fc = nn.Linear(in_channels, num_classes)
        if frozen:
            self.frozen()
        # post-process for inference
        post_process = getattr(self.criterion, "post_process", "none")
        if post_process == "softmax":
            self.post_process = nn.Softmax(dim=1)
        else:
            self.post_process = nn.Identity()

    def frozen(self):
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsMixupHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
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
        if self.fc is None:
            return x
        # test-time augmentation
        if len(x) > 1 and self.aug_test:
            aug_pred = [self.forward_head(_x, post_process) for _x in x]
            aug_pred = torch.stack(aug_pred).mean(dim=0)
            return [aug_pred]
        # simple test
        else:
            return [self.forward_head(x[0], post_process)]

    def lambda_adjust(self, lam, mode="pow", thr=1, idx=1):
        """ rescale lambda for two-hot label mixup classification
        
        Args:
            lam (float): The original lambda in [0,1].
            mode (str): The rescale function, {'pow', 'exp'}.
            thr (float): If lam < threshold, do rescale; else
                lam=1. Threshold in (0,1].
            idx (float): The index for power or exp functions.
        """
        if lam >= thr:
            lam = 1
        else:
            if mode == "pow":
                lam = (thr ** (-abs(idx))) * (lam ** abs(idx))
            elif mode == "exp":
                b = (abs(idx)** (-thr*2)) * 1
                k = 1 / (1 - b)
                lam = ((abs(idx)** (lam - thr*2)) * (abs(idx) ** lam) - b) * k
            else:
                raise NotImplementedError
        return lam

    def loss(self, cls_score, labels, label_mask=None, **kwargs):
        r"""" mixup classification loss forward
        
        Args:
            cls_score (list): Score should be [tensor] of [N, d].
            labels (tuple or tensor): Labels should be tensor [N, \*] by default.
                If labels as tuple, it's used for CE mixup, (gt_a, gt_b, lambda).
            label_mask (tensor): Mask (N,1) to indicate whether this idx is a
                ground truth or pseudo label.
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

        # 1. original one-hot classification
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
        # 2. mixup classification
        else:
            y_a, y_b, lam = labels
            if isinstance(lam, torch.Tensor):  # lam is scalar or tensor [N,1]
                lam = lam.unsqueeze(-1)
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                y_a.dim() == 1 or (y_a.dim() == 2 and y_a.shape[1] == 1)
            # Notice: we allow the single-label cls using multi-label loss, thus
            # * For single-label or multi-label cls, loss = loss.sum() / N
            avg_factor = y_a.size(0)
            
            # 2.1 mixup (hard ce) cls (using softmax)
            if not self.multi_label:
                assert self.two_hot == False
                losses['loss'] = \
                    self.criterion(cls_score, y_a, avg_factor=avg_factor, **kwargs) * lam + \
                    self.criterion(cls_score, y_b, avg_factor=avg_factor, **kwargs) * (1 - lam)
            else:
                # convert to onehot (binary) for multi-label mixup cls
                if single_label:
                    y_a = F.one_hot(y_a, num_classes=self.num_classes)
                    y_b = F.one_hot(y_b, num_classes=self.num_classes)
                # basic mixup labels: sumed to 1
                y_mixed = lam * y_a + (1 - lam) * y_b
                use_eta_weight = None
                class_weight = None

                # 2.2 mixup (sigmoid) multi-lalebl sumed to 2 (using two-hot loss)
                if self.two_hot:
                    if self.lam_scale_mode != 'none':
                        lam_a = self.lambda_adjust(
                            lam, mode=self.lam_scale_mode, thr=self.lam_thr, idx=self.lam_idx)
                        lam_b = self.lambda_adjust(
                            1-lam, mode=self.lam_scale_mode, thr=self.lam_thr, idx=self.lam_idx)
                        if label_mask is not None:
                            lam_a = lam_a if label_mask[0] else lam
                            lam_b = lam_b if label_mask[1] else 1-lam
                        y_mixed = lam_a * y_a + lam_b * y_b
                    else:
                        y_mixed = y_a + y_b
                # 2.3 mixup (soft) single-label sumed to 1 (using softmax)
                else:
                    if self.eta_weight["eta"] != 0:
                        # whether to use eta
                        below_thr = lam < self.eta_weight["thr"]
                        if self.eta_weight["mode"] == 'less':
                            use_eta_weight = [lam, 0] if below_thr else [0, 1-lam]
                        elif self.eta_weight["mode"] == 'more':
                            use_eta_weight = [lam, 0] if not below_thr else [0, 1-lam]
                        else:
                            use_eta_weight = [lam, 1-lam]  # 'both'
                        # eta rescale by lam
                        for i in range(len(use_eta_weight)):
                            if use_eta_weight[i] > 0:
                                if self.lam_scale_mode != 'none':
                                    use_eta_weight[i] = self.eta_weight["eta"] * \
                                        self.lambda_adjust(
                                            use_eta_weight[i], mode=self.lam_scale_mode,
                                            thr=self.lam_thr, idx=self.lam_idx)
                                else:
                                    use_eta_weight[i] = self.eta_weight["eta"]
                                assert use_eta_weight[0] > 0 or use_eta_weight[1] > 0, \
                                    "one of eta should be non-zero, lam={}, lam_={}".format(lam, 1-lam)
                # rescale the sum of labels, each hot <= 1
                if self.two_hot_scale < 1:
                    y_mixed = (y_mixed * self.two_hot_scale).clamp(max=1)
                # remove neg in BCE loss
                if self.neg_weight < 1:
                    class_weight = (y_mixed > 0).type(torch.float)
                    class_weight = class_weight.clamp(min=self.neg_weight)
                losses['loss'] = self.criterion(
                    cls_score, y_mixed,
                    avg_factor=avg_factor, class_weight_override=class_weight,
                    eta_weight=use_eta_weight, **kwargs)
            # compute accuracy
            losses['acc'] = accuracy(cls_score, labels[0])
            losses['acc_mix'] = accuracy_mixup(cls_score, labels)
        return losses
