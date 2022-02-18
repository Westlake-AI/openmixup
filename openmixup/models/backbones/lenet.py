# reference: https://github.com/open-mmlab/mmclassification/tree/master/mmcls/models/backbones
# copy from mmclassification lenet.py
import torch.nn as nn

from mmcv.cnn import kaiming_init, normal_init

from .. import builder
from ..registry import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class LeNet5(BaseBackbone):
    """`LeNet5 <https://en.wikipedia.org/wiki/LeNet>`_ backbone.
        12.29 version

    The input for LeNet-5 is a 32×32 grayscale image.

    Args:
        activation (str): choose your activation func, default is Tanh.
        mlp_neck (dict): additional MLP neck in SSL.
        cls_neck (dict): the original classifier MLP in LeNet,
            "120-tanh-84-tanh-class_num". Default is None.
    """

    def __init__(self,
                activation="Tanh",
                mlp_neck=None,
                cls_neck=None,
                pretrained=None):
        super(LeNet5, self).__init__()
        assert activation in ["ReLU", "LeakyReLU", "Tanh", "ELU", "Sigmoid"]
        assert mlp_neck is None or cls_neck is None
        self.activation = activation
        self.mlp_neck = mlp_neck
        self.cls_neck = cls_neck
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1), eval("nn.{}()".format(activation)),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1), eval("nn.{}()".format(activation)),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1), eval("nn.{}()".format(activation)),
        )
        if mlp_neck is not None:  # additional mlp neck for LeNet
            self.mlp_neck = builder.build_neck(mlp_neck)
        if cls_neck is not None:  # original cls neck in LeNet
            self.cls_neck = nn.Sequential(
                nn.Linear(120, 84),
                eval("nn.{}()".format(activation)),
                # nn.Linear(84, num_classes),  # ori LeNet
            )
        self.init_weights(pretrained=pretrained)
    
    def init_weights(self, pretrained=None):
        super(LeNet5, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    if self.activation not in ['LeakyReLU', "ReLU"]:
                        normal_init(m, std=0.01, bias=0.)
                    else:
                        kaiming_init(m, mode='fan_in', nonlinearity='relu')
            if self.mlp_neck is not None:
                self.mlp_neck.init_weights(init_linear='normal')
    
    def forward(self, x):
        x = self.features(x)
        if self.mlp_neck is not None:
            x = self.mlp_neck( [x.squeeze()] )[0]
        if self.cls_neck is not None:
            x = self.cls_neck(x.squeeze())

        return [x]
