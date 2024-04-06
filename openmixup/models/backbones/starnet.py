from typing import Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn.bricks import DropPath, build_norm_layer
from mmcv.runner.base_module import BaseModule

from .base_backbone import BaseBackbone
from ..builder import BACKBONES
from ..utils import to_2tuple


class ConvNorm(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes,
                 kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 with_bn=True, norm_cfg=dict(type='BN')):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', build_norm_layer(norm_cfg, out_planes)[1])
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(BaseModule):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg)

        self.dwconv = ConvNorm(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True, norm_cfg=norm_cfg)
        self.f1 = ConvNorm(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvNorm(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvNorm(mlp_ratio * dim, dim, 1, with_bn=True, norm_cfg=norm_cfg)
        self.dwconv2 = ConvNorm(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


@BACKBONES.register_module()
class StarNet(BaseBackbone):
    """StarNet.

    A PyTorch implementation of StarNet introduced by:
    `Rewrite the Stars <https://arxiv.org/abs/2403.19967>`_

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``TransNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.
            - mlp_ratios (list[int]): Expansion ratio of MLPs.

        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        drop_rate (float): Dropout rate. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        out_indices (Sequence | int): Output from which network position.
            Index 0-6 respectively corresponds to
            [stage1, downsampling, stage2, downsampling, stage3, downsampling, stage4]
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
        init_cfg (dict, optional): Initialization config dict
    """

    arch_settings = {
        's100': {
            'base_dim': 20,
            'depths': [1, 2, 4, 1],
        },
        's1': {
            'base_dim': 24,
            'depths': [2, 2, 8, 3],
        },
        's2': {
            'base_dim': 32,
            'depths': [1, 2, 6, 2],
        },
        's3': {
            'base_dim': 32,
            'depths': [2, 2, 8, 4],
        },
        's4': {
            'base_dim': 32,
            'depths': [3, 3, 12, 5],
        },
    }

    def __init__(self,
                 arch='s1',
                 in_channels=3,
                 mlp_ratio=4,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.,
                 out_indices=-1,
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'base_dim' in arch, \
                f'The arch dict must have "depths" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'
        depths = arch['depths']
        self.base_dim = arch['base_dim']
        self.num_stages = len(depths)

        # stem layer
        prev_dim = 32
        self.stem = nn.Sequential(
            ConvNorm(in_channels, prev_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(),
        )
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.stages = nn.ModuleList()
        cur = 0
        # build stages
        for i_layer in range(len(depths)):
            embed_dim = self.base_dim * 2 ** i_layer
            down_sampler = ConvNorm(prev_dim, embed_dim, 3, 2, 1)
            prev_dim = embed_dim
            blocks = [Block(prev_dim, mlp_ratio, dpr[cur + i], norm_cfg) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def init_weights(self, pretrained=None):
        super(StarNet, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is None:
                for m in self.modules():
                    if isinstance(m, (nn.Linear, nn.Conv2d)):
                        nn.init.trunc_normal_(m.weight, std=.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                        nn.init.zeros_(m.bias)
                        nn.init.ones_(m.weight)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.num_stages - 1:
                x = self.norm(x).mean(dim=[2, 3])
            if i in self.out_indices:
                outs.append(x)

        return outs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            self.stages[i].eval()
            for param in self.stages[i].parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(StarNet, self).train(mode)
        self._freeze_stages()
