# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
# modified from mmselfsup cae.py
import torch

from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class CAE(BaseModel):
    """CAE.

    Implementation of `Context Autoencoder for Self-Supervised Representation
    Learning <https://arxiv.org/abs/2202.03026>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of neck.
        head (dict): Config dict for module of head functions.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.0.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 init_cfg=None,
                 **kwargs):
        super(CAE, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)

        self.backbone = builder.build_backbone(backbone)
        self.teacher = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
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
        super(CAE, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)  # backbone
        for param_ol, param_tgt in zip(self.backbone.parameters(),
                                       self.teacher.parameters()):
            param_tgt.detach()
            param_tgt.requires_grad = False
            param_tgt.data.copy_(param_ol.data)

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the teacher network by hook."""
        for param_ol, param_tgt in zip(self.backbone.parameters(),
                                       self.teacher.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

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

        unmasked = self.backbone(img_v1, mask)
        # get the latent prediction for the masked patches
        with torch.no_grad():
            latent_target = self.teacher(img_v1, ~mask)
            latent_target = latent_target[:, 1:, :]
            self.momentum_update()

        pos_embed = self.backbone.pos_embed.expand(img.shape[0], -1, -1)
        pos_embed_masked = pos_embed[:, 1:][mask].reshape(
                                    img_v1.shape[0], -1, pos_embed.shape[-1])
        pos_embed_unmasked = pos_embed[:, 1:][~mask].reshape(
                                    img_v1.shape[0], -1, pos_embed.shape[-1])

        # input the unmasked tokens and masked tokens to the decoder
        logits, latent_pred = self.neck(unmasked[:, 1:], pos_embed_masked,
                                        pos_embed_unmasked)

        logits = logits.view(-1, logits.shape[-1])
        # inputs[1] is the target image
        loss_main, loss_align = self.head(logits, img_v2, latent_pred,
                                          latent_target, mask)
        losses = dict()
        losses['loss'] = loss_main + loss_align
        losses['main'] = loss_main
        losses['align'] = loss_align

        return losses
