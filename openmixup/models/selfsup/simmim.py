# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
# modified from mmselfsup simmim.py
import torch
import torch.nn.functional as F

from openmixup.utils import force_fp32, print_log
from ..classifiers import BaseModel
from ..utils import PlotTensor
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class SimMIM(BaseModel):
    """SimMIM.

    Implementation of `SimMIM: A Simple Framework for Masked Image Modeling
    <https://arxiv.org/abs/2111.09886>`_.

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
                 save=False,
                 init_cfg=None,
                 **kwargs):
        super(SimMIM, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)

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
        super(SimMIM, self).init_weights()

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

    @force_fp32(apply_to=('img_ori', 'img_rec', 'mask',))
    def plot_reconstruction(self, img_ori, img_rec, mask):
        """ visualize reconstruction results """
        mask = 1. - mask.unsqueeze(1).type_as(img_ori)
        mask = F.interpolate(mask, scale_factor=img_ori.size(2) / mask.size(2), mode="nearest")
        img_mask = img_ori * mask
        # plot MIM results
        img = torch.cat((img_ori[:4], img_mask[:4], img_rec[:4]), dim=0)
        assert self.save_name.find(".png") != -1
        self.ploter.plot(img, nrow=4, title_name="SimMIM", save_name=self.save_name)

    def forward_train(self, img, **kwargs):
        """Forward the masked image and get the reconstruction loss.

        Args:
            x (List[torch.Tensor, torch.Tensor]): Images and masks.

        Returns:
            dict: Reconstructed loss.
        """
        mask = kwargs.get('mask', None)
        if isinstance(img, list):
            img, mask = img
        if isinstance(mask, list):
            mask, _ = mask

        img_latent = self.backbone(img, mask)
        img_rec = self.neck(img_latent)
        if isinstance(img_rec, list):
            img_rec = img_rec[-1]
        losses = self.head(img, img_rec, mask)

        if self.save:
            self.plot_reconstruction(img, img_rec, mask)
        
        return losses
