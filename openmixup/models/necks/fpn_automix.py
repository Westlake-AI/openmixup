import torch.nn as nn
from ..registry import NECKS
from mmcv.cnn import ConvModule


@NECKS.register_module()
class FPN_AutoMix(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 act_cfg=None):
        super(FPN_AutoMix, self).__init__()
        self.l_conv = ConvModule(
                in_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                inplace=False)

    def forward(self, input):
        '''
        input: feature of two layers, 0 for target
        '''
        assert len(input) == 2
        n, c, w, h = input[0].shape # target shape
        
        if w > input[1].shape[-1]:
            # upsample
            m = nn.Upsample(scale_factor=2, mode='nearest')
            out = m(input[-1])
        else:
            # avgpool
            m = nn.AdaptiveAvgPool2d((w, h))
            out = m(input[-1])
        last_feature = self.l_conv(out)
        out_feature = input[0] + last_feature
        
        return out_feature
        