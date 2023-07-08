import torch
import torch.nn as nn

from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class activation(nn.ReLU):
    """ Activation in VanillaNet

    Series informed activation function. Implemented by conv.
    """

    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.act_num = act_num
        self.deploy = deploy
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        if deploy:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        trunc_normal_init(self.weight, mean=0, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x), 
                self.weight, self.bias, padding=self.act_num, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class Block(nn.Module):
    """Network Block in VanillaNet"""

    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False, pool_type="maxpool", ada_pool=None):
        super().__init__()
        self.act_learn = 1
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        if pool_type == 'ada' or ada_pool is not None:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))
        elif pool_type == 'avgpool':
            self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(stride)
        elif pool_type == 'conv':
            self.pool = nn.Identity() if stride == 1 else nn.Conv2d(
                dim_out, dim_out, kernel_size=stride, stride=stride, padding=0)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)

        self.act = activation(dim_out, act_num, deploy=self.deploy)
 
    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            # We use leakyrelu to implement the deep training technique.
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
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
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True


@BACKBONES.register_module()
class VanillaNet(BaseBackbone):
    r""" VanillaNet
        A PyTorch impl of : `VanillaNet: the Power of Minimalism in Deep Learning` -
        <https://arxiv.org/abs/2305.12972>`_

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``VanillaNet.arch_settings``. And if dict, it
            should include the following three keys:

            - embed_dims (list[int]): The number of channels at each stage.
            - strides (list[int]): The downsampling stride number of each stage.

            Defaults to 'vanillanet_9'.

        in_channels (int): Number of input image channels. Defaults to 3.
        act_num (int): Number of layers in the activation. Defaults to 3.
        deploy (bool): Whether to switch the model structure to deployment
            mode. Default: False.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
    """
    arch_settings = {
        'vanillanet_5': {
            'embed_dims': [128*4, 256*4, 512*4, 1024*4],
            'strides': [2, 2, 2],
        },
        'vanillanet_6': {
            'embed_dims': [128*4, 256*4, 512*4, 1024*4, 1024*4],
            'strides': [2, 2, 2, 1],
        },
        'vanillanet_7': {
            'embed_dims': [128*4, 128*4, 256*4, 512*4, 1024*4, 1024*4],
            'strides': [1, 2, 2, 2, 1],
        },
        'vanillanet_8': {
            'embed_dims': [128*4, 128*4, 256*4, 512*4, 512*4, 1024*4, 1024*4],
            'strides': [1, 2, 2, 1, 2, 1],
        },
        'vanillanet_9': {
            'embed_dims': [128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
            'strides': [1,2,2,1,1,2,1],
        },
        'vanillanet_10': {
            'embed_dims': [128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
            'strides': [1,2,2,1,1,1,2,1],
        },
        'vanillanet_11': {
            'embed_dims': [128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
            'strides': [1,2,2,1,1,1,1,2,1],
        },
        'vanillanet_12': {
            'embed_dims': [128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
            'strides': [1,2,2,1,1,1,1,1,2,1],
        },
        'vanillanet_13': {
            'embed_dims': [128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
            'strides': [1,2,2,1,1,1,1,1,1,2,1],
        },
        'vanillanet_13_x1_5': {
            'embed_dims': [128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
            'strides': [1,2,2,1,1,1,1,1,1,2,1],
        },
        'vanillanet_13_x1_5_ada_pool': {
            'embed_dims': [128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
            'strides': [1,2,2,1,1,1,1,1,1,2,1],
            'ada_pool': [0,38,19,0,0,0,0,0,0,10,0],
        },
    }

    def __init__(self,
                 arch='vanillanet_9',
                 in_channels=3,
                 in_patch_size=4,
                 in_stride=4,
                 in_pad=0,
                 num_classes=None,
                 drop_rate=0,
                 act_num=3,
                 deploy=False,
                 ada_pool=None,
                 pool_type="maxpool",
                 act_learn_init=None,
                 act_learn_invert=False,
                 out_indices=-1,
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'embed_dims' in arch and 'strides' in arch, \
                f'The arch dict must have "embed_dims" and "strides", ' \
                f'but got {list(arch.keys())}.'

        embed_dims = arch['embed_dims']
        strides = arch['strides']
        ada_pool = ada_pool if ada_pool is not None else arch.get('ada_pool', None)
        assert len(embed_dims) == len(strides) + 1
        self.num_stage = self.depth = len(strides)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, list), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stage + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.num_classes = num_classes
        self.deploy = deploy
        patch_size, stride, padding = (in_patch_size, in_stride, in_pad) \
            if not ada_pool else (4, 3, 1)

        if self.deploy:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims[0], kernel_size=patch_size, stride=stride, padding=padding),
                activation(embed_dims[0], act_num, deploy=self.deploy)
            )
        else:
            self.stem1 = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims[0], kernel_size=patch_size, stride=stride, padding=padding),
                nn.BatchNorm2d(embed_dims[0], eps=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(embed_dims[0], eps=1e-6),
                activation(embed_dims[0], act_num)
            )

        self.act_learn = 1  # modified during training

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = Block(dim=embed_dims[i], dim_out=embed_dims[i+1], act_num=act_num,
                              stride=strides[i], deploy=deploy, pool_type=pool_type)
            else:
                stage = Block(dim=embed_dims[i], dim_out=embed_dims[i+1], act_num=act_num,
                              stride=strides[i], deploy=deploy, pool_type=pool_type, ada_pool=ada_pool[i])
            self.stages.append(stage)

        if num_classes is not None:
            if self.deploy:
                self.cls = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Dropout(drop_rate),
                    nn.Conv2d(embed_dims[-1], num_classes, 1),
                )
            else:
                self.cls1 = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Dropout(drop_rate),
                    nn.Conv2d(embed_dims[-1], num_classes, 1),
                    nn.BatchNorm2d(num_classes, eps=1e-6),
                )
                self.cls2 = nn.Sequential(
                    nn.Conv2d(num_classes, num_classes, 1)
                )

        self._freeze_stages()
        self.act_learn_invert = act_learn_invert
        if act_learn_init is not None:
            assert isinstance(act_learn_init, float) and 0 <= act_learn_init <= 1
            self.change_act(act_learn_init)

    def init_weights(self, pretrained=None):
        super(VanillaNet, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_init(m, mean=0., std=0.02, bias=0)
                elif isinstance(m, (nn.LayerNorm, _BatchNorm, nn.GroupNorm)):
                    constant_init(m, val=1, bias=0)

    def update_attribute(self, attribute):
        """Interface for updating the attribute in the backbone"""
        self.change_act(attribute)

    def change_act(self, m):
        if self.act_learn_invert:
            m = 1 - m
        for i in range(self.num_stage):
            self.stages[i].act_learn = m
        self.act_learn = m

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deploy:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                for j in range(2):
                    stem = getattr(self, f'stem{j}')
                    stem.eval()
                    for param in stem.parameters():
                        param.requires_grad = False

        for i in range(self.frozen_stages):
            self.stages[i].eval()
            for param in self.stages[i].parameters():
                param.requires_grad = False

        if self.num_classes and self.frozen_stages == self.num_stage:
            if self.deploy:
                self.cls.eval()
                for param in self.cls.parameters():
                    param.requires_grad = False
            else:
                for j in range(2):
                    cls = getattr(self, f'cls{j}')
                    cls.eval()
                    for param in cls.parameters():
                        param.requires_grad = False

    def forward(self, x):
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem1(x)
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            x = self.stem2(x)

        outs = []
        for i in range(self.num_stage):
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(x)

        if self.num_classes is not None:
            if self.deploy:
                x = self.cls(x)
            else:
                x = self.cls1(x)
                x = torch.nn.functional.leaky_relu(x,self.act_learn)
                x = self.cls2(x)
            x = x.view(x.size(0), -1)
            outs.append(x)

        return outs

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

    def switch_to_deploy(self):
        self.stem2[2].switch_to_deploy()
        kernel, bias = self._fuse_bn_tensor(self.stem1[0], self.stem1[1])
        self.stem1[0].weight.data = kernel
        self.stem1[0].bias.data = bias
        kernel, bias = self._fuse_bn_tensor(self.stem2[0], self.stem2[1])
        self.stem1[0].weight.data = torch.einsum(
            'oi,icjk->ocjk', kernel.squeeze(3).squeeze(2), self.stem1[0].weight.data)
        self.stem1[0].bias.data = bias + (self.stem1[0].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.stem = torch.nn.Sequential(*[self.stem1[0], self.stem2[2]])
        self.__delattr__('stem1')
        self.__delattr__('stem2')

        for i in range(self.num_stage):
            self.stages[i].switch_to_deploy()

        if self.num_classes is not None:
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

    def train(self, mode=True):
        super(VanillaNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
