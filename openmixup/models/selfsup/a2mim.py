import torch
import torch.nn.functional as F
from openmixup.utils import force_fp32, print_log

from ..classifiers import BaseModel
from ..utils import PlotTensor
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class A2MIM(BaseModel):
    """A^2MIM.

    Implementation of `Architecture-Agnostic Masked Image Modeling--From ViT back to CNN
    <https://arxiv.org/abs/2205.13943>`_.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        save (bool): Saving reconstruction results. Defaults to False.
        init_cfg (dict, optional): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 residual=False,
                 save=False,
                 init_cfg=None,
                 **kwargs):
        super(A2MIM, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)

        self.residual = residual
        self.save = save
        self.save_name = 'reconstruction'
        self.ploter = PlotTensor(apply_inv=True)
        
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(A2MIM, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()

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

    @force_fp32(apply_to=('img', 'img_mim', 'img_rec', 'mask',))
    def plot_reconstruction(self, img, img_mim, img_rec, mask):
        """ visualize reconstruction results """
        nrow = 4
        img_mim = img_mim[:nrow]
        img_rec = img_rec[:nrow]
        img = img[:nrow]
        img_raw = None
        plot_args = dict(dpi=None, apply_inv=True)
        
        mask = 1. - mask[:4].unsqueeze(1).type_as(img_mim)
        mask = F.interpolate(mask, scale_factor=img_mim.size(2) / mask.size(2), mode="nearest")
        img_mask = img_mim * mask
        if img_raw is not None:
            img = torch.cat((img_raw, img_mim, img_mask, img_rec), dim=0)
        else:
            img = torch.cat((img_mim, img_mask, img_rec), dim=0)
        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=nrow, title_name="A2MIM", save_name=self.save_name, **plot_args)

    def forward_train(self, img, **kwargs):
        """Forward the masked image and get the reconstruction loss.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
            mask (torch.Tensor): MIM mask of shape (N, H, W).

        Returns:
            dict: Reconstructed loss.
        """
        # raw img and MIM targets
        mask = kwargs.get('mask', None)
        if isinstance(mask, list):
            mask, img_mim = mask
        else:
            img_mim = img.clone()

        img_latent = self.backbone(img, mask)
        img_rec = self.neck(img_latent)
        if isinstance(img_rec, list):
            img_rec = img_rec[-1]
        if self.residual:
            img_rec += img_mim.mean(dim=(2, 3), keepdim=True).expand(img_rec.size())
        losses = self.head(img_mim, img_rec, mask)

        if self.save:
            self.plot_reconstruction(img, img_mim, img_rec, mask)
        
        return losses
