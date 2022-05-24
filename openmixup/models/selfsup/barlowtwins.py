# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
# modified from mmselfsup barlowtwins.py
from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class BarlowTwins(BaseModel):
    """BarlowTwins.

    Implementation of `Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction <https://arxiv.org/abs/2103.03230>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/barlowtwins/blob/main/main.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone. Defaults to None.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(BarlowTwins, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(BarlowTwins, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)  # backbone
        self.neck.init_weights(init_linear='kaiming')

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

        z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        losses = self.head(z1, z2)['loss']
        return dict(loss=losses)
