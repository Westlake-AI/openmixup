# reference: https://github.com/open-mmlab/mmclassification/tree/master/mmcls/models/backbones
# copy from mmclassification alexnet.py
import torch.nn as nn

from mmcv.cnn import kaiming_init, normal_init

from .. import builder
from ..registry import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class AlexNet(BaseBackbone):
    """`AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_ backbone.

    The input for AlexNet is a 224x224 RGB image.

    Args:
        mlp_neck (dict): additional MLP neck in SSL. Default is None.
        cls_neck (dict): the original classifier MLP in AlexNet,
            "Dropout-fc-ReLU-Dropout-fc-ReLU-4096-class_num".
    """

    def __init__(self,
                mlp_neck=None,
                cls_neck=None,
                pretrained=None):
        super(AlexNet, self).__init__()
        assert mlp_neck is None or cls_neck is None
        self.mlp_neck = mlp_neck
        self.cls_neck = cls_neck
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
        if cls_neck is not None:  # original cls neck in AlexNet
            self.cls_neck = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                # nn.Linear(4096, num_classes),  # ori
            )
        self.init_weights(pretrained=pretrained)
    
    def init_weights(self, pretrained=None):
        super(AlexNet, self).init_weights(pretrained)
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        if self.mlp_neck is not None:
            self.mlp_neck.init_weights(init_linear='normal')
        if self.cls_neck is not None:
            for m in self.cls_neck:
                if isinstance(m, nn.Linear):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        if self.mlp_neck is not None:
            x = [x.view(x.size(0), 256 * 6 * 6)]
            x = self.mlp_neck(x)[0]
        if self.cls_neck is not None:
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.cls_neck(x)
        
        return [x]
