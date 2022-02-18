from packaging import version
import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, normal_init

from ..registry import NECKS
from ..utils import build_norm_layer


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


@NECKS.register_module
class AvgPoolNeck(nn.Module):
    """Average pooling neck.
    """

    def __init__(self):
        super(AvgPoolNeck, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        return [self.avg_pool(x[0])]


@NECKS.register_module
class LinearNeck(nn.Module):
    """Linear neck: fc only.
    """

    def __init__(self, in_channels, out_channels, with_avg_pool=True):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.fc(x.view(x.size(0), -1))]


@NECKS.register_module
class RelativeLocNeck(nn.Module):
    """Relative patch location neck: fc-bn-relu-dropout.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(RelativeLocNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc = nn.Linear(in_channels * 2, out_channels)
        if sync_bn:
            _, self.bn = build_norm_layer(
                dict(type='SyncBN', momentum=0.003),
                out_channels)
        else:
            self.bn = nn.BatchNorm1d(
                out_channels, momentum=0.003)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear, std=0.005, bias=0.1)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn, x)
        else:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return [x]


@NECKS.register_module
class ODCNeck(nn.Module):
    """The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(ODCNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc0 = nn.Linear(in_channels, hid_channels)
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN', momentum=0.001, affine=False),
                hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(
                hid_channels, momentum=0.001, affine=False)

        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]


@NECKS.register_module
class MoCoV2Neck(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
        v12.29: add activation choices.

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
                 activation="ReLU",  # add 12.29
                 with_avg_pool=True):
        super(MoCoV2Neck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if activation != "ReLU":
            assert activation in ["LeakyReLU", "Tanh", "ELU", "Sigmoid"]
            if activation == "Tanh" or activation == "Sigmoid":  # add 12.30
                inplace_choice = ""
            else:
                inplace_choice = "inplace=True"
            self.mlp = nn.Sequential(  # using my activation func
                nn.Linear(in_channels, hid_channels), eval( "nn.{}({})".format(activation, inplace_choice) ),
                nn.Linear(hid_channels, out_channels))
        else:  # ori in MoCo.v2 (ReLU)
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
                nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module()
class NonLinearNeck(nn.Module):
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
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
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
                 norm_cfg=dict(type='SyncBN'),
                ):
        super(NonLinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x = bn(x)
        return [x]


@NECKS.register_module()
class SwAVNeck(nn.Module):
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
                ):
        super(SwAVNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.with_l2norm = with_l2norm
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if out_channels == 0:
            self.projection_neck = None
        elif hid_channels == 0:
            self.projection_neck = nn.Linear(in_channels, out_channels)
        else:
            self.bn = build_norm_layer(norm_cfg, hid_channels)[1]
            self.projection_neck = nn.Sequential(
                nn.Linear(in_channels, hid_channels), self.bn,
                nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

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
            _x = _x[0]
            if self.with_avg_pool:
                _out = self.avgpool(_x)
                avg_out.append(_out)
        feat_vec = torch.cat(avg_out)  # [sum(num_crops) * N, C]
        feat_vec = feat_vec.view(feat_vec.size(0), -1)
        output = self.forward_projection(feat_vec)
        return [output]


@NECKS.register_module()
class DenseCLNeck(nn.Module):
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
                ):
        super(DenseCLNeck, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = True if num_grid is not None else False
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        """Forward function of neck.

        Args:
            x (list[tensor]): feature map of backbone.
        """
        assert len(x) == 1
        x = x[0]

        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            x = self.pool(x)  # sxs
        x = self.mlp2(x)  # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x)  # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1)  # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1)  # bxd
        return [avgpooled_x, x, avgpooled_x2]
