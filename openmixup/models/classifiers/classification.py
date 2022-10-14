from openmixup.utils import print_log

from .base_model import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class Classification(BaseModel):
    """Simple image classification and regression.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Default: None.
        head (dict): Config dict for module of classification or regression
            loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(Classification, self).__init__(init_cfg, **kwargs)
        self.backbone = builder.build_backbone(backbone)
        assert isinstance(head, dict)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
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
        if self.with_neck:
            self.neck.init_weights(init_linear='kaiming')
        self.head.init_weights()

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        outs = self.head(x)
        losses = self.head.loss(outs, gt_label)
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.backbone(img)  # tuple
        if self.with_neck:
            x = self.neck(x)
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def augment_test(self, img):
        """Test function with test time augmentation."""
        x = list()
        for _img in img:
            if self.with_neck:
                x.append(self.neck(self.backbone(_img))[0])
            else:
                x.append(self.backbone(_img)[0])
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_test(self, img, **kwargs):
        """
        Args:
            img (List[Tensor] or Tensor): the outer list indicates the
                test-time augmentations and inner Tensor should have a
                shape of (N, C, H, W).
        """
        if isinstance(img, list):
            return self.augment_test(img)
        else:
            return self.simple_test(img)
