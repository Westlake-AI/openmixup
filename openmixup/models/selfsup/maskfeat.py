import numpy as np
import torch
import torch.nn.functional as F
from openmixup.utils import force_fp32, print_log
from openmixup.models.utils import Canny, HOG, Laplacian, Sobel

from ..classifiers import BaseModel
from ..utils import PlotTensor
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class MaskFeat(BaseModel):
    """MaskFeat.

    Implementation of `Masked Feature Prediction for Self-Supervised
    Visual Pre-Training <https://arxiv.org/abs/2112.09133>`_.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        backbone_k (dict): Config dict for pre-trained backbone. Default: None.
        mim_target (None or str): Mode of MIM target. Notice that 'HOG' is
            borrowed as SlowFast implementation (9*12) while 'hog' is implemented
            by scimage (9). Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        pretrained_k (str, optional): Path to pre-trained weights for backbone_k,
            e.g., DINO or CLIP pre-training.
            Default: None.
        save (bool): Saving reconstruction results. Defaults to False.
        init_cfg (dict, optional): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 backbone_k=None,
                 mim_target="HOG",
                 residual=False,
                 pretrained=None,
                 pretrained_k=None,
                 save=False,
                 init_cfg=None,
                 **kwargs):
        super(MaskFeat, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.backbone_k = None
        if backbone_k is not None:
            self.backbone_k = builder.build_backbone(backbone_k)
            for param in self.backbone_k.parameters():  # stop grad k
                param.requires_grad = False

        # mim targets
        self.mim_target = mim_target
        self.residual = residual
        assert self.mim_target in [
            None, 'gray', 'canny', 'hog', 'HOG', 'laplacian', 'lbp', 'pretrained', 'sobel',]
        if self.mim_target == 'canny':
            self.feat_layer = Canny(non_max_suppression=True, edge_smooth=True)
        elif self.mim_target == 'HOG':
            self.feat_layer = HOG(nbins=9, pool=8, gaussian_window=16)
        elif self.mim_target == 'laplacian':
            self.feat_layer = Laplacian(mode='DoG', use_threshold=False)
        elif self.mim_target == 'sobel':
            self.feat_layer = Sobel(isotropic=True, use_threshold=True, out_channels=2)

        self.save = save
        self.save_name = 'reconstruction'
        self.ploter = PlotTensor(apply_inv=True)
        
        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights. Default: None.
            pretrained_k (str, optional): Path to pre-trained weights for encoder_k.
                Default: None.
        """
        super(MaskFeat, self).init_weights()

        # init pre-trained params
        if pretrained_k is not None:
            print_log('load pre-training from: {}'.format(pretrained_k), logger='root')
            if self.backbone_k is not None:
                self.backbone_k.init_weights(pretrained=pretrained_k)
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        if self.backbone_k is not None and pretrained_k is None:
            for param_q, param_k in zip(self.backbone.parameters(),
                                        self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)

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
        # plot MIM results
        if self.mim_target == 'hog':
            from ..utils import hog_visualization
            plot_args = dict(dpi=400, apply_inv=False)
            hog_img, hog_rec = list(), list()
            orientations = img_mim.size(1)
            pixels_per_cell = (img.size(2) // img_mim.size(2), img.size(3) // img_mim.size(3))
            img_mim = img_mim.permute(0, 2, 3, 1)
            img_rec = img_rec.permute(0, 2, 3, 1)
            for i in range(nrow):
                hog_img.append(
                    hog_visualization(img_mim[i], img.shape[2:], orientations, pixels_per_cell))
                hog_rec.append(
                    hog_visualization(img_rec[i], img.shape[2:], orientations, pixels_per_cell))
            img_mim = torch.from_numpy(np.array(hog_img))
            img_raw = img.mean(dim=1, keepdim=True).detach().cpu()
            img_rec = torch.from_numpy(np.array(hog_rec))
        
        if img_mim.size(1) not in [1, 3]:
            print_log(f"Warning: the shape of img_mim is invalid={img_mim.shape}")
            img_mim = img_mim.mean(dim=1, keepdim=True)
        if img_rec.size(1) not in [1, 3]:
            print_log(f"Warning: the shape of img_rec is invalid={img_rec.shape}")
            img_rec = img_rec.mean(dim=1, keepdim=True)
        mask = 1. - mask[:4].unsqueeze(1).type_as(img_mim)
        mask = F.interpolate(mask, scale_factor=img_mim.size(2) / mask.size(2), mode="nearest")
        img_mask = img_mim * mask

        if img_raw is not None:
            img = torch.cat((img_raw, img_mim, img_mask, img_rec), dim=0)
        else:
            img = torch.cat((img_mim, img_mask, img_rec), dim=0)
        assert self.save_name.find(".png") != -1
        feat_name = self.mim_target if self.mim_target is not None else 'rgb'
        self.ploter.plot(
            img, nrow=nrow, title_name="MaskFeat_"+feat_name, save_name=self.save_name, **plot_args)

    def forward_train(self, img, **kwargs):
        """Forward the masked image and get the reconstruction loss.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
            mask (torch.Tensor or list): MIM mask of shape (N, H, W).

        Returns:
            dict: Reconstructed loss.
        """
        # raw img and MIM targets
        mask = kwargs.get('mask', None)
        if isinstance(mask, list):
            mask, img_mim = mask
        else:
            if isinstance(img, list):
                img, mask = img
            img_mim = img.clone()

        if self.mim_target in ['canny', 'HOG', 'laplacian', 'sobel',]:
            assert img_mim.size(1) == 3
            img_mim = self.feat_layer(img_mim)
        elif self.mim_target == 'gray':
            img_mim = img_mim.mean(dim=1, keepdim=True)
        elif self.mim_target == 'pretrained':
            img_mim = self.backbone_k(img_mim)[-1]

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
