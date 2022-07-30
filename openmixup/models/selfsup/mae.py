# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
# modified from mmselfsup mae.py
import torch

from openmixup.utils import force_fp32, print_log
from ..classifiers import BaseModel
from ..utils import PlotTensor
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    
    Args:
        backbone (dict): Config dict for encoder.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        save (bool): Saving reconstruction results. Defaults to False.
        init_cfg (dict): Config dict for weight initialization.
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
        super(MAE, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
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
        super(MAE, self).init_weights()

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
        # unpatchify reconstructed imgs
        img_rec = self.head.unpatchify(img_rec)
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 * 3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        img_mask = img_ori * (1 - mask)

        # plot MIM results
        img = torch.cat((img_ori[:4], img_mask[:4], img_rec[:4]), dim=0)
        assert self.save_name.find(".png") != -1
        self.ploter.plot(img, nrow=4, title_name="MAE", save_name=self.save_name)

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        latent, mask, ids_restore = self.backbone(img)
        pred = self.neck(latent, ids_restore)
        if isinstance(pred, list):
            pred = pred[-1]
        losses = self.head(img, pred, mask)

        if self.save:
            self.plot_reconstruction(img, pred, mask)

        return losses
