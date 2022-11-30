# reference: https://github.com/open-mmlab/mmclassification/tree/master/mmcls/models/backbones
# copy from mmclassification alexnet.py
import torch.nn as nn

from mmcv.cnn import kaiming_init

from .. import builder
from ..registry import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class AlexNet(BaseBackbone):
    """`AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_ backbone.

    ImageNet classification with deep convolutional neural networks
    <https://dl.acm.org/doi/10.1145/3065386>.

    The input for AlexNet is a 224x224 RGB image.

    Args:
        num_classes (int): The number of categroies. Defaults to 1000.
        cls_head (dict): the original classifier MLP in AlexNet,
            "Dropout-fc-ReLU-Dropout-fc-ReLU-4096-class_num".
        mlp_neck (dict): additional MLP neck in SSL. Default is None.
    """

    def __init__(self,
                 num_classes=1000,
                 cls_head=True,
                 mlp_neck=None,
                 pretrained=None):
        super(AlexNet, self).__init__()
        assert mlp_neck is None or cls_head is True
        self.num_classes = num_classes
        self.mlp_neck = None
        self.cls_head = None
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if mlp_neck is not None:  # additional mlp neck for AlexNet
            self.mlp_neck = builder.build_neck(mlp_neck)
        if cls_head:  # original cls neck in AlexNet
            assert isinstance(num_classes, int)
            self.cls_head = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(AlexNet, self).init_weights(pretrained)
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        if self.mlp_neck is not None:
            self.mlp_neck.init_weights(init_linear='normal')
        if self.cls_head is not None:
            for m in self.cls_head:
                if isinstance(m, nn.Linear):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        if self.mlp_neck is not None:
            x = [x.view(x.size(0), 256 * 6 * 6)]
            x = self.mlp_neck(x)[0]
        if self.cls_head is not None:
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.cls_head(x)
        return [x]

    def train(self, mode=True):
        super(AlexNet, self).train(mode)
