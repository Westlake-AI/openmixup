import torch
import torch.nn.functional as F
import torch.nn as nn
from torch._six import inf

from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn import kaiming_init, normal_init


class GradWeighter(nn.Module):
    def __init__(self, in_channels, mode):
        super(GradWeighter, self).__init__()
        assert mode in ['plain', 'concat', 'minmax', 'softmax']
        self.in_channels = in_channels
        self.mode = mode
        if mode == 'concat':
            self.concat_project = ConvModule(
                self.in_channels * 2,
                self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ReLU', inplace=False))
        self.init_weights()
        
    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.mode == 'concat':
            assert init_linear in ['normal', 'kaiming'], \
                "Undefined init_linear: {}".format(init_linear)
            if init_linear == 'normal':
                    normal_init(self.concat_project.conv, std=std)
            else:
                kaiming_init(self.concat_project.conv, mode='fan_in', nonlinearity='relu')
        else:
            pass

    def forward(self, feature, grad):
        """ different weighting methods """
        assert feature.shape == grad.shape
        if self.mode == 'plain':
            weight = torch.mean(grad, axis=(2,3), keepdim=True)
            weighted_feature = feature * weight 

        elif self.mode == 'concat':
            concat_feature = torch.cat([feature, grad], dim=1)
            weighted_feature = self.concat_project(concat_feature) + feature

        elif self.mode == 'softmax':
            weight = torch.mean(grad, axis=(2,3), keepdim=True)
            softmax_weight = F.softmax(weight, dim=1)
            weighted_feature = feature * softmax_weight

        elif self.mode == 'minmax':
            weight = torch.mean(grad, axis=(2,3), keepdim=True)
            weight -= weight.min(1, keepdim=True)[0]
            weight /= weight.max(1, keepdim=True)[0]
            weighted_feature = feature * weight

        return weighted_feature


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    total_norm = 0
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
