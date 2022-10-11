from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class BEiT(BaseModel):
    """BEiT.

    Implementation of `BEiT: BERT Pre-Training of Image Transformers
     <https://arxiv.org/abs/2106.08254>`_.

    Args:
        backbone (dict, optional): Config dict for module of backbone.
        neck (dict, optional): Config dict for module of deep features to
            compact feature vectors. Defaults to None.
        head (dict, optional): Config dict for module of loss functions.
            Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(BEiT, self).__init__(init_cfg, **kwargs)
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
        super(BEiT, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs of (N, D).
        """
        x = self.backbone(img)
        if len(x) == 3:
            # return cls_token, yeilding better performances than patch tokens
            x = x[0][:, 0]
        else:
            x = x[0][-1]  # return cls_token
        return [x]

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (List[torch.Tensor]): List of [img_1, img_2, mask].
                The img is a list of two views with shape (N, C, H, W).

        Returns:
            dict: Reconstructed loss.
        """
        assert isinstance(img, list) and len(img) == 3
        img_v1 = img[0].contiguous()
        img_v2 = img[1].contiguous()
        mask = img[2].contiguous()

        img_latent = self.backbone(img_v1, mask)
        logits = self.neck(img_latent[:, 1:, :])
        loss = self.head(logits, img_v2, mask)
        losses = dict(loss=loss)

        return losses
