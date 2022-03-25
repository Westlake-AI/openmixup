import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init, ConvModule
from ..registry import NECKS


@NECKS.register_module
class ConvNeck(nn.Module):
    """The N layers conv neck: [conv-norm-act] - conv-{norm}.

    Args:
        in_channels (int): Channels of the input feature map.
        hid_channels (int): Channels of the hidden feature channel.
        out_channels (int): Channels of the output feature channel.
        num_layers (int): The number of convolution layers.
        kernel_size (int): Kernel size of the convolution layer.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        with_bias (bool): Whether to use 'bias' in each conv layer. Default: False.
        with_avg_pool (bool): Whether to add a global average pooling layer in the
            output. Default: False.
        with_last_norm (bool): Whether to add a norm layer in the output. Default: False.
        with_last_dropout (float or dict): Probability of an element to be zeroed in
            the output, or dict config for dropout.
            Default: 0.0.
        with_residual (bool, optional): Add resudual connection.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 kernel_size=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ELU'),
                 with_bias=False,
                 with_avg_pool=False,
                 with_last_norm=False,
                 with_last_dropout=0.,
                 with_residual=True,
                 **kwargs):
        super(ConvNeck, self).__init__()
        # basic args
        self.in_channels = int(in_channels)
        self.hid_channels = int(hid_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        assert kernel_size in [1, 3]
        # specific for ssl
        self.with_bias = bool(with_bias)
        self.with_avg_pool = bool(with_avg_pool)
        self.with_last_norm = bool(with_last_norm)
        self.with_residual = bool(with_residual)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) \
            if with_avg_pool else nn.Identity()
        if isinstance(with_last_dropout, dict):
            _type = with_last_dropout.pop('type', None)
            _prob = with_last_dropout.pop('prob', 0.)
            assert 0 < _prob and _prob < 1 and \
                _type in ["Dropout", "AlphaDropout", "FeatureAlphaDropout"]
            self.dropout = eval("nn.{}".format(_type))(_prob)
        elif float(with_last_dropout) > 0:
            assert float(with_last_dropout) < 1.
            self.dropout = nn.Dropout(float(with_last_dropout))
        else:
            self.dropout = nn.Identity()

        # build FFN
        layers = []
        for i in range(num_layers):
            layers.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else hid_channels,
                    out_channels=hid_channels if i != num_layers-1 else out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1 if kernel_size == 3 else 0,
                    bias=with_bias,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if i != num_layers-1 or with_last_norm else None,
                    act_cfg=act_cfg if i != num_layers-1 else None
                ))
        self.conv = nn.Sequential(*layers)
        self.init_weights()
    
    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        res = x[0]
        x = self.dropout(self.conv(x[0]))
        if self.with_avg_pool:
            x = self.avgpool(x).view(x.size(0), -1)
        if self.with_residual:
            x = x + res
        return [x]
