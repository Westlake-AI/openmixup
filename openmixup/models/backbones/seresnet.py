# reference: https://github.com/open-mmlab/mmclassification/tree/master/mmcls/models/backbones
# copy from mmclassification seresnet.py
import mmcv
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
import torch.utils.checkpoint as cp

from ..registry import BACKBONES
from .resnet_mmcls import Bottleneck, ResLayer, ResNet


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class SEBottleneck(Bottleneck):
    """SEBottleneck block for SEResNet.

    Args:
        in_channels (int): The input channels of the SEBottleneck block.
        out_channels (int): The output channel of the SEBottleneck block.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16
    """

    def __init__(self, in_channels, out_channels, se_ratio=16, **kwargs):
        super(SEBottleneck, self).__init__(in_channels, out_channels, **kwargs)
        self.se_layer = SELayer(out_channels, ratio=se_ratio)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.se_layer(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class SEGroupBottleneck(SEBottleneck):
    """SEBottleneck block for SEResNeXt.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        base_channels (int): Middle channels of the first stage. Default: 64.
        groups (int): Groups of conv2.
        width_per_group (int): Width per group of conv2. 64x4d indicates
            ``groups=64, width_per_group=4`` and 32x8d indicates
            ``groups=32, width_per_group=8``.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None
        se_ratio (int): Squeeze ratio in SELayer. Default: 16
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=64,
                 groups=32,
                 width_per_group=4,
                 se_ratio=16,
                 **kwargs):
        super(SEGroupBottleneck, self).__init__(in_channels, out_channels, se_ratio,
                                                **kwargs)
        self.groups = groups
        self.width_per_group = width_per_group

        # We follow the same rational of ResNext to compute mid_channels.
        # For SEResNet bottleneck, middle channels are determined by expansion
        # and out_channels, but for SEResNeXt bottleneck, it is determined by
        # groups and width_per_group and the stage it is located in.
        if groups != 1:
            assert self.mid_channels % base_channels == 0
            self.mid_channels = (
                groups * width_per_group * self.mid_channels // base_channels)

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)


@BACKBONES.register_module()
class SEResNet(ResNet):
    """SEResNet backbone.

    Pytorch implementation of `Squeeze-and-Excitation Networks
    <https://arxiv.org/abs/1709.01507>`_

    Args:
        depth (int): Network depth, from {50, 101, 152}.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmcls.models import SEResNet
        >>> import torch
        >>> self = SEResNet(depth=50)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 56, 56)
        (1, 128, 28, 28)
        (1, 256, 14, 14)
        (1, 512, 7, 7)
    """

    arch_settings = {
        50: (SEBottleneck, (3, 4, 6, 3)),
        101: (SEBottleneck, (3, 4, 23, 3)),
        152: (SEBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, se_ratio=16, **kwargs):
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for SEResNet')
        self.se_ratio = se_ratio
        super(SEResNet, self).__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(se_ratio=self.se_ratio, **kwargs)


@BACKBONES.register_module()
class SEResNet_CIFAR(SEResNet):
    """SEResNet backbone for CIFAR.

    Compared to standard ResNeXt, it uses `kernel_size=3` and `stride=1` in
    conv1, and does not apply MaxPoolinng after stem. It has been proven to
    be more efficient than standard ResNet in other public codebase, e.g.,
    `https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py`.

    """

    def __init__(self, depth, deep_stem=False, **kwargs):
        super(SEResNet_CIFAR, self).__init__(
            depth, deep_stem=deep_stem, **kwargs)
        assert not self.deep_stem, 'SEResNet_CIFAR do not support deep_stem'

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs
        return outs


@BACKBONES.register_module()
class SEResNeXt(SEResNet):
    """SEResNeXt backbone.

    Pytorch implementation of `Squeeze-and-Excitation Networks
    <https://arxiv.org/abs/1709.01507>`_

    Args:
        depth (int): Network depth, from {50, 101, 152}.
        groups (int): Groups of conv2 in Bottleneck. Default: 32.
        width_per_group (int): Width per group of conv2 in Bottleneck.
            Default: 4.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    """

    arch_settings = {
        50: (SEGroupBottleneck, (3, 4, 6, 3)),
        101: (SEGroupBottleneck, (3, 4, 23, 3)),
        152: (SEGroupBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, groups=32, width_per_group=4, **kwargs):
        self.groups = groups
        self.width_per_group = width_per_group
        super(SEResNeXt, self).__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(
            groups=self.groups,
            width_per_group=self.width_per_group,
            base_channels=self.base_channels,
            **kwargs)
