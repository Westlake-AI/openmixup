import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init, normal_init, trunc_normal_init

from openmixup.models.backbones.metaformer import SquaredReLU
from .cls_head import BaseClsHead
from ..registry import HEADS


@HEADS.register_module
class EfficientFormerClsHead(BaseClsHead):
    """EfficientFormer classifier head.

    A PyTorch implementation of EfficientFormer head: `EfficientFormer:
    Vision Transformers at MobileNet Speed <https://arxiv.org/abs/2206.01191>`_

    """

    def __init__(self, distillation=True, **kwargs):
        super(EfficientFormerClsHead, self).__init__(**kwargs)
        # build a classification head
        self.dist = distillation
        assert self.hidden_dim is None
        if self.num_classes <= 0 or self.num_classes is None:
            raise ValueError(
                f'num_classes={self.num_classes} must be a positive integer')

        self.head = nn.Linear(self.in_channels, self.num_classes)
        if self.dist:
            self.dist_head = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        self.head.eval()
        for param in self.head.parameters():
            param.requires_grad = False
        if self.dist:
            self.dist_head.eval()
            for param in self.dist_head.parameters():
                param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(EfficientFormerClsHead, self).init_weights()
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
        cls_score = self.head(x)
        if self.dist:
            cls_score = (cls_score + self.dist_head(x)) / 2

        if post_process:
            cls_score = self.post_process(cls_score)
        return cls_score


@HEADS.register_module
class MetaFormerClsHead(BaseClsHead):
    """MetaFormer Baselines classifier head.

    A PyTorch implementation of Mlp heads in : `MetaFormer Baselines for Vision` -
        <https://arxiv.org/abs/2210.13452>`_

    """

    def __init__(self, mlp_ratio=4, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True, **kwargs):
        super(MetaFormerClsHead, self).__init__(**kwargs)

        # build a classification head
        self.hidden_dim = int(mlp_ratio * self.in_channels)
        self.fc1 = nn.Linear(self.in_channels, self.hidden_dim, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

        if self.frozen:
            self._freeze()

    def _freeze(self):
        for head in ["fc1", "norm", "fc2"]:
            m = getattr(self, head)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(MetaFormerClsHead, self).init_weights()
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
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)

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
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        if post_process:
            x = self.post_process(x)
        return x
