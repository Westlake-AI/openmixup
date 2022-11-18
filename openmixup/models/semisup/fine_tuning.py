import torch.nn as nn
from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class FineTuning(BaseModel):
    """Vanilla image classification.

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 head=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(FineTuning, self).__init__(init_cfg, **kwargs)
        self.backbone = builder.build_backbone(backbone)
        if head is not None:
            self.head = builder.build_head(head)
        else:
            self.head = nn.Identity()
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        if self.head is not None:
            self.head.init_weights()

    def forward_train(self, img, gt_labels, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_labels (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input both must have 5 dims, got: {}".format(img.dim())
        img = img[:, 0, ...].contiguous()
        x = self.forward_backbone(img)
        outs = self.head(x)
        losses = self.head.loss(outs, gt_labels)
        return losses

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))
