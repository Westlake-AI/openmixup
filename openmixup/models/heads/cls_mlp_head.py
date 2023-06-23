from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, constant_init, kaiming_init,
                      normal_init, trunc_normal_init)
from mmcv.runner.base_module import BaseModule, ModuleList

from openmixup.models.backbones.metaformer import SquaredReLU
from .cls_head import BaseClsHead
from ..registry import HEADS
from ..utils import build_norm_layer


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


class BatchNormLinear(BaseModule):

    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN1d')):
        super(BatchNormLinear, self).__init__()
        self.bn = build_norm_layer(norm_cfg, in_channels)
        self.linear = nn.Linear(in_channels, out_channels)

    @torch.no_grad()
    def fuse(self):
        w = self.bn.weight / (self.bn.running_var + self.bn.eps)**0.5
        b = self.bn.bias - self.bn.running_mean * \
            self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
        w = self.linear.weight * w[None, :]
        b = (self.linear.weight @ b[:, None]).view(-1) + self.linear.bias

        self.linear.weight.data.copy_(w)
        self.linear.bias.data.copy_(b)
        return self.linear

    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        return x


def fuse_parameters(module):
    for child_name, child in module.named_children():
        if hasattr(child, 'fuse'):
            setattr(module, child_name, child.fuse())
        else:
            fuse_parameters(child)


@HEADS.register_module
class LeViTClsHead(BaseClsHead):
    """MetaFormer Baselines classifier head.

    A PyTorch implementation of Mlp heads in : `MetaFormer Baselines for Vision` -
        <https://arxiv.org/abs/2210.13452>`_

    """

    def __init__(self, distillation=False, deploy=False, **kwargs):
        super(LeViTClsHead, self).__init__(**kwargs)

        # build a classification head
        self.deploy = deploy
        self.distillation = distillation
        self.head = BatchNormLinear(self.in_channels, self.num_classes)
        if distillation:
            self.head_dist = BatchNormLinear(self.in_channels, self.num_classes)

        if self.deploy:
            self.switch_to_deploy()

        if self.frozen:
            self._freeze()

    def switch_to_deploy(self):
        if self.deploy:
            return
        fuse_parameters(self)
        self.deploy = True

    def _freeze(self):
        head_list = ["head", "head_dist"] if self.distillation else ["head",]
        for head in head_list:
            m = getattr(self, head)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(LeViTClsHead, self).init_weights()
            return
        assert init_linear in ['normal', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
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
        if self.distillation:
            x = self.head(x), self.head_dist(x)  # 2 16 384 -> 2 1000
            if not self.training:
                x = (x[0] + x[1]) / 2
            else:
                raise NotImplementedError("OpenMixup doesn't support "
                                          'training in distillation mode.')
        else:
            x = self.head(x)
        if post_process:
            x = self.post_process(x)
        return x


class LinearBlock(BaseModule):
    """Linear block for StackedLinearClsHead."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate=0.,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fc = nn.Linear(in_channels, out_channels)

        self.norm = None
        self.act = None
        self.dropout = None

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        if act_cfg is not None:
            self.act = build_activation_layer(act_cfg)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """The forward process."""
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


@HEADS.register_module
class StackedLinearClsHead(BaseClsHead):
    """Classifier head with several hidden fc layer and a output fc layer.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        mid_channels (Sequence[int]): Number of channels in the hidden fc
            layers.
        dropout_rate (float): Dropout rate after each hidden fc layer,
            except the last layer. Defaults to 0.
        norm_cfg (dict, optional): Config dict of normalization layer after
            each hidden fc layer, except the last layer. Defaults to None.
        act_cfg (dict, optional): Config dict of activation function after each
            hidden layer, except the last layer. Defaults to use "ReLU".
    """

    def __init__(self,
                 mid_channels: Sequence[int],
                 dropout_rate=0.,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(StackedLinearClsHead, self).__init__(**kwargs)

        # build a classification head
        assert isinstance(mid_channels, Sequence), \
            f'`mid_channels` of StackedLinearClsHead should be a sequence, ' \
            f'instead of {type(mid_channels)}'
        self.mid_channels = mid_channels
        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self._init_layers()

        if self.frozen:
            self._freeze()

    def _init_layers(self):
        """"Init layers."""
        self.layers = ModuleList()
        in_channels = self.in_channels
        for hidden_channels in self.mid_channels:
            self.layers.append(
                LinearBlock(
                    in_channels,
                    hidden_channels,
                    dropout_rate=self.dropout_rate,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            in_channels = hidden_channels

        self.layers.append(
            LinearBlock(
                self.mid_channels[-1],
                self.num_classes,
                dropout_rate=0.,
                norm_cfg=None,
                act_cfg=None))

    def _freeze(self):
        self.layers.eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def pre_logits(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``x`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

    @property
    def fc(self):
        """Full connected layer."""
        return self.layers[-1]

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(StackedLinearClsHead, self).init_weights()
            return
        assert init_linear in ['normal', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)

    def forward_head(self, x, post_process=False):
        """The forward process."""
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        pre_logits = self.pre_logits(x)
        x = self.fc(pre_logits)
        if post_process:
            x = self.post_process(x)
        return x


@HEADS.register_module
class VanillaNetClsHead(BaseClsHead):
    """VanillaNet classifier head.

    A PyTorch impl of : `VanillaNet: the Power of Minimalism in Deep Learning` -
        <https://arxiv.org/abs/2305.12972>`_

    """

    def __init__(self, drop_rate=0., deploy=False, **kwargs):
        super(VanillaNetClsHead, self).__init__(**kwargs)
        self.deploy = deploy
        self.act_learn = 1  # modified during training
        assert not self.with_avg_pool

        # build a classification head
        if self.deploy:
            self.cls = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(self.in_channels, self.num_classes, 1),
            )
        else:
            self.cls1 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(self.in_channels, self.num_classes, 1),
                nn.BatchNorm2d(self.num_classes, eps=1e-6),
            )
            self.cls2 = nn.Sequential(
                nn.Conv2d(self.num_classes, self.num_classes, 1)
            )

        if self.frozen:
            self._freeze()

    def update_attribute(self, attribute):
        """Interface for updating the attribute in the head"""
        self.act_learn = attribute

    def _freeze(self):
        head_list = ["cls1", "cls2"] if not self.deploy else ['cls']
        for head in head_list:
            m = getattr(self, head)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(VanillaNetClsHead, self).init_weights()
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
        if self.deploy:
            x = self.cls(x)
        else:
            x = self.cls1(x)
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            x = self.cls2(x)
        x = x.view(x.size(0), -1)
        if post_process:
            x = self.post_process(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    @torch.no_grad()
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.cls1[2], self.cls1[3])
        self.cls1[2].weight.data = kernel
        self.cls1[2].bias.data = bias
        kernel, bias = self.cls2[0].weight.data, self.cls2[0].bias.data
        self.cls1[2].weight.data = torch.matmul(
            kernel.transpose(1, 3), self.cls1[2].weight.data.squeeze(3).squeeze(2)).transpose(1, 3)
        self.cls1[2].bias.data = bias + (self.cls1[2].bias.data.view(1, -1, 1, 1)*kernel).sum(3).sum(2).sum(1)
        self.cls = torch.nn.Sequential(*self.cls1[0:3])
        self.__delattr__('cls1')
        self.__delattr__('cls2')
        self.deploy = True
