import torch
import numpy as np
import torchvision
import os
import matplotlib.pyplot as plt
from torchvision import transforms

from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS
from ..augments import cutmix, fmix, gridmix, mixup, resizemix, saliencymix, smoothmix
from ..utils import (GatherLayer, batch_shuffle_ddp, batch_unshuffle_ddp)


@MODELS.register_module
class SimCLRMix(BaseModel):
    """SimCLR mixup baseline V0913 (update 09.17)

    Implementation of "A Simple Framework for Contrastive Learning
    of Visual Representations (https://arxiv.org/abs/2002.05709)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        alpha (int): Beta distribution '$\beta(\alpha, \alpha)$'.
        mix_block (dict or str): Config dict for the mixblock, or mixup mode.
            If using AutoMix, the mix_block should be a head config dict.
            else, using various mixup methods.
        mix_args (dict): Args for ResizeMix, FMix mode.
        head_weights (dict): Dict of the ssl heads names and loss weights.
            Default AutoMix: dict(head_ssl=1, head_mix=1, head_mix_block=1)
            Default Mixup: dict(head_ssl=1, head_mix=1)
        cross_view_gen (bool): Whether to generate mixup samples with (im_q,im_k), or
            with (im_q,im_q) as the same view.
            Default: False. (using the same view)
        cross_view_ssl (bool): Whether to caculate ssl loss for mixup samples with
            cross view embeddings when mixup samples are built by mixing im_q and im_k',
            i.e., lam * pos_logits(mix, k) + (1-lam) * pos_logits(mix, q').
            Default: False. (using the same view)
    """

    def __init__(self,
                backbone,
                neck=None,
                head=None,
                pretrained=None,
                alpha=2,
                mix_block=None,
                mix_args=dict(
                    manifoldmix=dict(layer=(0, 3)),
                    resizemix=dict(scope=(0.1, 0.8)),
                    fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False),
                ),
                head_weights=dict(
                    head_ssl=1, head_mix=1, head_mix_block=1),
                cross_view_gen=False,
                cross_view_ssl=False,
                save=False,
                save_name='MixedSamples',
                init_cfg=None,
                **kwargs):
        super(SimCLRMix, self).__init__(init_cfg, **kwargs)
        self.backbone = builder.build_backbone(backbone)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.mix_block = mix_block
        if mix_block is not None:
            if isinstance(mix_block, dict):
                self.mix_block = builder.build_head(mix_block)
            else:
                assert isinstance(mix_args, dict)
                assert mix_block in [
                    "mixup", "manifoldmix", "cutmix", "saliencymix", "resizemix", "fmix"]
                if mix_block in ["manifoldmix"]:
                    assert 0 == min(mix_args[mix_block]["layer"]) and max(mix_args[mix_block]["layer"]) < 4
                if mix_block == "resizemix":
                    assert 0 <= min(mix_args[mix_block]["scope"]) and max(mix_args[mix_block]["scope"]) <= 1
        else:
            raise NotImplementedError
        # ssl and mixup heads
        self.weight_ssl = head_weights.get("head_ssl", 1.)
        self.weight_mix = head_weights.get("head_mix", 1.)
        # basic params
        self.alpha = alpha
        self.mix_args = mix_args
        self.cross_view_gen = cross_view_gen
        self.cross_view_ssl = cross_view_ssl
        self.save = save
        self.save_name = save_name

    @staticmethod
    def _create_buffer(N):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(SimCLRMix, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
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
        im_q = img[0].contiguous()
        im_k = img[1].contiguous()
        losses = dict()
        
        # =============== Step 1: SimCLR forward ================
        x = self.forward_backbone(img.reshape(
            img.size(0) * 2, img.size(2), img.size(3), img.size(4)))  # 2nxCxHxW
        z = self.neck(x)[0]  # (2n)xd
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        loss_simclr = self.forward_SimCLR_loss(z)

        # =============== Step 2: Mixup forward =================
        if self.mix_block not in ["manifoldmix"]:
            gt_label = None  # no use in ssl
            if self.cross_view_gen:
                im_mixed = img.clone()  # [N, 2, C, H, W]
            else:
                im_mixed = im_q.clone()
            if self.mix_block not in ["fmix", "resizemix"]:
                mix_args = dict(dist_mode=True, alpha=self.alpha)
            else:
                mix_args = dict(dist_mode=True, alpha=self.alpha, **self.mix_args[self.mix_block])
            # call mixup methods
            im_mixed, gt_label = eval(self.mix_block)(im_mixed, gt_label, **mix_args)
            idx_shuffle_mix, _, lam = gt_label  # update lam and mixup shuffle index
            if self.save and self.cross_view_gen:  # show mixup samples
                im_, _, _ = batch_shuffle_ddp(im_k, idx_shuffle=idx_shuffle_mix)
                self.plot_mix(im_mixed, im_q, im_, lam, name=self.mix_block)
            
            # mixup forward
            mix = self.forward_backbone(im_mixed)
            mix = self.neck(mix)[0]  # nxd
            mix = mix / (torch.norm(mix, p=2, dim=1, keepdim=True) + 1e-10)
            if self.cross_view_gen:
                im_mixed = img.reshape(  # [N, 2, C, H, W] -> [2N, C, H, W]
                    img.size(0) * 2, img.size(2), img.size(3), img.size(4)).clone()
        else:
            im_mixed = im_q.clone()
            # manifoldmix
            lam = np.random.beta(self.alpha, self.alpha)
            _, idx_shuffle_mix, _ = batch_shuffle_ddp(im_q, no_repeat=True)  # N index, to shuffle im_k' or im_q'
            _layer = np.random.randint(  # [0, 3)
                min(self.mix_args[self.mix_block]["layer"]), max(self.mix_args[self.mix_block]["layer"]), dtype=int)
            # generate mask in terms of _layer
            _mask = None
            if img.size(3) > 64:  # normal version of resnet
                scale_factor = 2**(1 + _layer) if _layer > 0 else 1
            else:  # CIFAR version
                scale_factor = 2**(_layer - 1) if _layer > 1 else 1
            _mask_size = im_q.size(3) // scale_factor
            _mask = torch.zeros(im_q.size(0), 1, _mask_size, _mask_size).cuda()
            _mask[:] = lam

            # args for backbone
            mix_args = dict(
                layer=_layer, cross_view=self.cross_view_gen, mask=_mask,
                BN_shuffle=False, idx_shuffle_BN=None, idx_shuffle_mix=idx_shuffle_mix, dist_shuffle=True)
            mix = self.backbone(im_mixed, mix_args)[-1]
            mix = self.neck_q([mix])[0]  # mix: N
            mix = mix / (torch.norm(mix, p=2, dim=1, keepdim=True) + 1e-10)

        # method notice: assuming cross_view_gen, i.e., im_mixed = im_q + im_k (mixup shuffle)
        # cross_view_ssl is True: we caculate the lambda postive pair of mix and detach_k, and the
        #   1-lambda positive pair of mix and detach_q (mixup shuffled).
        N = z.size(0) // 2
        z_q = z[:N, :]  # using the first way (q) to caculate the mixup loss
        z_k = z[N:, :]
        z  = z_q if not self.cross_view_ssl else z_k
        z_ = z_q if self.cross_view_gen == self.cross_view_ssl else z_k

        loss_mix = \
            self.forward_SimCLR_loss(torch.cat([mix, z ]))["loss"] * lam + \
            self.forward_SimCLR_loss(torch.cat([mix, z_]))["loss"] * (1-lam)
        
        losses["loss"] = loss_simclr['loss'] * self.weight_ssl + loss_mix * self.weight_mix
        return losses
    
    def forward_SimCLR_loss(self, z):
        """ original SimCLR loss forward based on two way features z """
        assert z.dim() == 2
        # z is alreadly L2-normalized and 2Nxd
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        return losses

    @torch.no_grad()
    def plot_mix(self, im_mixed, im, im_, lam, name=""):
        invTrans = transforms.Compose([
            transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1/0.2023, 1/0.1994, 1/0.201]),
            transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[ 1., 1., 1. ])])
        imgs = torch.cat((im[:4], im_[:4], im_mixed[:4]), dim=0)
        img_grid = torchvision.utils.make_grid(imgs, nrow=4, pad_value=0)
        imgs = np.transpose(invTrans(img_grid).detach().cpu().numpy(), (1, 2, 0))
        fig = plt.figure()
        plt.imshow(imgs)
        plt.title('lambda {}: {}'.format(name, lam))
        if not os.path.exists(self.save_name):
            plt.savefig(self.save_name)
        plt.close()
