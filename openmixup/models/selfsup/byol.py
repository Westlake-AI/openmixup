# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
# modified from mmselfsup byol.py
import torch
import torch.nn as nn

from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class BYOL(BaseModel):
    """BYOL.

    Implementation of "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning (https://arxiv.org/abs/2006.07733)".

    *** Requiring Hook: `momentum_update` is adjusted by `CosineScheduleHook`
        after_train_iter in `momentum_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target network.
            Default: 0.996.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 init_cfg=None,
                 **kwargs):
        super(BYOL, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        
        self.backbone = self.online_net[0]
        self.neck = self.online_net[1]
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(BYOL, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained) # backbone
        self.online_net[1].init_weights(init_linear='kaiming') # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.requires_grad = False
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.head.init_weights(init_linear='normal')

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network by hook."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list) and len(img) >= 2
        img_v1 = img[0].contiguous()
        img_v2 = img[1].contiguous()
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        # compute target features
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0].detach()
            proj_target_v2 = self.target_net(img_v2)[0].detach()

        losses = \
            self.head(proj_online_v1, proj_target_v2)['loss'] + \
            self.head(proj_online_v2, proj_target_v1)['loss']
        return dict(loss=losses)
