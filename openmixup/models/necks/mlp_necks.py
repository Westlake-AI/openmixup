import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (build_norm_layer,
                      constant_init, kaiming_init, normal_init)
from mmcv.runner import BaseModule

from ..registry import NECKS


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
        elif isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            constant_init(m, val=1, bias=0)


@NECKS.register_module()
class GeneralizedMeanPooling(BaseModule):
    """Generalized Mean Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        p (float): Parameter value. Default: 3.
        eps (float): Epsilon. Default: 1e-6
        clamp (bool): Use clamp before pooling. Default: True
    """

    def __init__(self, p=3., eps=1e-6, clamp=True):
        assert p >= 1, "'p' must be a value greater then 1"
        super(GeneralizedMeanPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.clamp = clamp

    def gmp(self, x, p, eps=1e-6, clamp=True):
        if clamp:
            x = x.clamp(min=eps)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def forward(self, x):
        assert len(x) == 1
        outs = self.gmp(x, p=self.p, eps=self.eps, clamp=self.clamp)
        outs = outs.view(x.size(0), -1)
        return [outs]


@NECKS.register_module
class AvgPoolNeck(BaseModule):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(AvgPoolNeck, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        return [self.avg_pool(x[-1])]


@NECKS.register_module
class MaskPoolNeck(BaseModule):
    """Average pooling with mask."""

    def __init__(self, use_mask=True, output_size=1):
        super(MaskPoolNeck, self).__init__()
        self.use_mask = use_mask
        self.avg_pool = nn.AdaptiveAvgPool2d((output_size, output_size))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x, mask=None):
        assert len(x) == 1
        if mask is None or not self.use_mask:
            return [self.avg_pool(x[-1])]
        else:
            x = x[-1]
            B, _, H, W = x.size()
            if mask.shape[2] > H:
                mask_h = mask.shape[2]
                assert mask_h % H == 0
                mask = mask.view(B, 1, mask_h, -1).type_as(x)
                mask = F.upsample(mask, scale_factor=H / mask_h, mode="nearest")
            else:
                mask = mask.view(B, 1, H, W).type_as(x)
            # mask should have non-zero elements
            x = (x * mask).mean(dim=[2, 3]) / mask.mean(dim=[2, 3])
            return [x]


@NECKS.register_module
class LinearNeck(BaseModule):
    """The linear neck: fc only.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_avg_pool=True,
                 init_cfg=None):
        super(LinearNeck, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) \
            if with_avg_pool else nn.Identity()
        self.fc = nn.Linear(in_channels, out_channels)

    def init_weights(self, init_linear='normal', **kwargs):
        if self.init_cfg is not None:
            super(LinearNeck, self).init_weights()
        else:
            _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[-1]
        x = self.avgpool(x).view(x.size(0), -1)
        return [self.fc(x)]


@NECKS.register_module
class RelativeLocNeck(BaseModule):
    """The neck of relative patch location: fc-bn-relu-dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN1d').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_avg_pool=True,
                 norm_cfg=dict(type='BN1d'),
                 init_cfg=None):
        super(RelativeLocNeck, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) \
            if with_avg_pool else nn.Identity()
        self.fc = nn.Linear(in_channels * 2, out_channels)
        self.bn = build_norm_layer(
            dict(**norm_cfg, momentum=0.003), out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

    def init_weights(self, init_linear='normal', **kwargs):
        if self.init_cfg is not None:
            super(RelativeLocNeck, self).init_weights()
        else:
            _init_weights(self, init_linear, std=0.005, bias=0.1)

    def forward(self, x):
        assert len(x) == 1
        x = x[-1]
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return [x]


@NECKS.register_module
class ODCNeck(BaseModule):
    """The non-linear neck of ODC: fc-bn-relu-dropout-fc-relu.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 norm_cfg=dict(type='SyncBN'),
                 with_avg_pool=True,
                 init_cfg=None):
        super(ODCNeck, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) \
            if with_avg_pool else nn.Identity()
        self.fc0 = nn.Linear(in_channels, hid_channels)
        self.bn0 = build_norm_layer(
            dict(**norm_cfg, momentum=0.001, affine=False), hid_channels)[1]
        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

    def init_weights(self, init_linear='normal', **kwargs):
        if self.init_cfg is not None:
            super(ODCNeck, self).init_weights()
        else:
            _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[-1]
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]


@NECKS.register_module
class MoCoV2Neck(BaseModule):
    """The non-linear neck in MoCo v2: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 init_cfg=None):
        super(MoCoV2Neck, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) \
            if with_avg_pool else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal', **kwargs):
        if self.init_cfg is not None:
            super(MoCoV2Neck, self).init_weights()
        else:
            _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[-1]
        x = self.avgpool(x).view(x.size(0), -1)
        return [self.mlp(x)]


@NECKS.register_module()
class NonLinearNeck(BaseModule):
    """The non-linear neck for SimCLR and BYOL.

    Structure: fc-bn-[relu-fc-bn] where the substructure in [] can be repeated.
    For the default setting, the repeated time is 1.
    The neck can be used in many algorithms, e.g., SimCLR, BYOL, SimSiam.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam and BarlowTwins).
            Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
        vit_backbone (bool): Whether to use ViT (use cls_token) backbones. The
            cls_token will be removed in this neck. Defaults to False.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_bias=False,
                 with_last_bn=True,
                 with_last_bn_affine=True,
                 with_last_bias=False,
                 with_avg_pool=True,
                 vit_backbone=False,
                 norm_cfg=dict(type='SyncBN'),
                 init_cfg=None):
        super(NonLinearNeck, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) \
            if with_avg_pool else nn.Identity()
        self.vit_backbone = vit_backbone
        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = build_norm_layer(norm_cfg, hid_channels)[1]

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(hid_channels, this_channels, bias=with_bias))
                self.add_module(f'bn{i}',
                                build_norm_layer(norm_cfg, this_channels)[1])
                self.bn_names.append(f'bn{i}')
            else:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(
                        hid_channels, this_channels, bias=with_last_bias))
                if with_last_bn:
                    self.add_module(
                        f'bn{i}',
                        build_norm_layer(
                            dict(**norm_cfg, affine=with_last_bn_affine),
                            this_channels)[1])
                    self.bn_names.append(f'bn{i}')
                else:
                    self.bn_names.append(None)
            self.fc_names.append(f'fc{i}')

    def init_weights(self, init_linear='normal', **kwargs):
        if self.init_cfg is not None:
            super(NonLinearNeck, self).init_weights()
        else:
            _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[-1]
        if self.vit_backbone:  # remove cls_token
            x = x[-1]
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.bn0(self.fc0(x))
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = fc(self.relu(x))
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x = bn(x)
        return [x]


@NECKS.register_module()
class SwAVNeck(BaseModule):
    """The non-linear neck of SwAV: fc-bn-relu-fc-normalization.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global average pooling after
            backbone. Defaults to True.
        with_l2norm (bool): whether to normalize the output after projection.
            Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 with_l2norm=True,
                 norm_cfg=dict(type='SyncBN'),
                 init_cfg=None):
        super(SwAVNeck, self).__init__(init_cfg)
        self.with_l2norm = with_l2norm
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) \
            if with_avg_pool else nn.Identity()
        if out_channels == 0:
            self.projection_neck = None
        elif hid_channels == 0:
            self.projection_neck = nn.Linear(in_channels, out_channels)
        else:
            self.bn = build_norm_layer(norm_cfg, hid_channels)[1]
            self.projection_neck = nn.Sequential(
                nn.Linear(in_channels, hid_channels), self.bn,
                nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal', **kwargs):
        if self.init_cfg is not None:
            super(SwAVNeck, self).init_weights()
        else:
            _init_weights(self, init_linear, **kwargs)

    def forward_projection(self, x):
        if self.projection_neck is not None:
            x = self.projection_neck(x)
        if self.with_l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x

    def forward(self, x):
        # forward computing
        # x: list of feature maps, len(x) according to len(num_crops)
        avg_out = []
        for _x in x:
            _x = _x[-1]
            avg_out.append(self.avgpool(_x))
        feat_vec = torch.cat(avg_out)  # [sum(num_crops) * N, C]
        feat_vec = feat_vec.view(feat_vec.size(0), -1)
        output = self.forward_projection(feat_vec)
        return [output]


@NECKS.register_module()
class DenseCLNeck(BaseModule):
    """The non-linear neck of DenseCL.

    Single and dense neck in parallel: fc-relu-fc, conv-relu-conv.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL`_.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_grid (int): The grid size of dense features. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None,
                 init_cfg=None):
        super(DenseCLNeck, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = True if num_grid is not None else False
        self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid)) \
            if self.with_pool else nn.Identity()
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal', **kwargs):
        if self.init_cfg is not None:
            super(DenseCLNeck, self).init_weights()
        else:
            _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        """Forward function of neck.

        Args:
            x (list[tensor]): feature map of backbone.
        """
        assert len(x) == 1
        x = x[-1]
        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        x = self.mlp2(self.pool(x))  # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x)  # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1)  # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1)  # bxd
        return [avgpooled_x, x, avgpooled_x2]
