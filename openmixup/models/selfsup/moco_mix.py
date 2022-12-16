import torch
import torch.nn as nn
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
from ..utils import (concat_all_gather, batch_shuffle_ddp, batch_unshuffle_ddp, \
                     grad_batch_shuffle_ddp, grad_batch_unshuffle_ddp)


@MODELS.register_module
class MoCoMix(BaseModel):
    """MOCO mixup baseline V0721 (fixed by 09.13)

    Implementation of "Momentum Contrast for Unsupervised Visual
    Representation Learning (https://arxiv.org/abs/1911.05722)".
    Part of the code is borrowed from:
    "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        queue_len (int): Number of negative keys maintained in the queue.
            Default: 65536.
        feat_dim (int): Dimension of compact feature vectors. Default: 128.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
        alpha (int): Beta distribution '$\beta(\alpha, \alpha)$'.
        mix_block (str): Name denotes the mixup mode for various mixup methods.
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
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 alpha=2,
                 mask_loss=0.,
                 mix_block="mixup",
                 mix_args=dict(
                    manifoldmix=dict(layer=(0, 3)),
                    resizemix=dict(scope=(0.1, 0.8), use_alpha=False),
                    fmix=dict(decay_power=3, size=(32, 32), max_soft=0., reformulate=False),
                 ),
                 head_weights=dict(head_ssl=1, head_mix=1),
                 cross_view_gen=False,
                 cross_view_ssl=False,
                 save=False,
                 save_name="MixedSamples",
                 init_cfg=None,
                 **kwargs):
        super(MoCoMix, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.encoder_q = builder.build_backbone(backbone)
        self.encoder_k = builder.build_backbone(backbone)
        self.neck_q = builder.build_neck(neck)
        self.neck_k = builder.build_neck(neck)
        self.head = builder.build_head(head)

        self.mix_block = mix_block
        if mix_block is not None:
            if isinstance(mix_block, dict):
                raise NotImplementedError
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
        
        self.init_weights(pretrained=pretrained)
        # basic params
        self.queue_len = queue_len
        self.momentum = momentum
        self.alpha = alpha
        self.mask_loss = mask_loss
        self.mix_args = mix_args
        self.cross_view_gen = cross_view_gen
        self.cross_view_ssl = cross_view_ssl
        self.save = save
        self.save_name = save_name

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(MoCoMix, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q.init_weights(pretrained=pretrained)
        self.neck_q.init_weights(init_linear='kaiming')
        # stop grad for k
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.neck_q.parameters(),
                                    self.neck_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.neck_q.parameters(),
                                    self.neck_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

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
        # various mixup methods
        if isinstance(self.mix_block, str):
            # =============== Step 1: MOCO forward ================
            loss_moco, detach_k, detach_q = self.forward_moco(im_q, im_k)

            # =============== Step 2: Mixup forward ===============
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
                
                # mixup forward shuffle for BN
                im_mixed, _, idx_unshuffle = grad_batch_shuffle_ddp(im_mixed)
                mix = self.encoder_q(im_mixed)[-1]
                mix = self.neck_q([mix])[0]  # mix q: NxC
                mix = nn.functional.normalize(mix, dim=1)
                # undo forward shuffle
                mix = grad_batch_unshuffle_ddp(mix, idx_unshuffle)
            
            else:
                if self.cross_view_gen:
                    im_mixed = img.reshape(  # [N, 2, C, H, W] -> [2N, C, H, W]
                        img.size(0) * 2, img.size(2), img.size(3), img.size(4)).clone()
                else:
                    im_mixed = im_q.clone()
                # manifoldmix
                lam = np.random.beta(self.alpha, self.alpha)
                _, idx_shuffle_mix, _ = batch_shuffle_ddp(im_q, no_repeat=True)  # N index, to shuffle im_k' or im_q'
                # _layer = np.random.randint(0, 3, dtype=int)  # [0, 3)
                _layer = np.random.randint(
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

                # mixup forward shuffle for BN
                _, idx_shuffle_BN, idx_unshuffle_BN = batch_shuffle_ddp(im_q)  # N index for shuffle_BN
                # args for backbone
                mix_args = dict(
                    layer=_layer, cross_view=self.cross_view_gen, mask=_mask,
                    BN_shuffle=True, idx_shuffle_BN=idx_shuffle_BN,
                    idx_shuffle_mix=idx_shuffle_mix, dist_shuffle=True)
                mix = self.encoder_q(im_mixed, mix_args)[-1]
                mix = self.neck_q([mix])[0]  # mix q: NxC
                mix = nn.functional.normalize(mix, dim=1)
                # undo forward shuffle
                mix = grad_batch_unshuffle_ddp(mix, idx_unshuffle_BN)  # N index

            # method notice: assuming cross_view_gen, i.e., im_mixed = im_q + im_k (mixup shuffle)
            # cross_view_ssl is True: we caculate the lambda postive pair of mix and detach_k, and the
            #   1-lambda positive pair of mix and detach_q (mixup shuffled).
            detach  = detach_q if not self.cross_view_ssl else detach_k
            detach_ = detach_q if self.cross_view_gen == self.cross_view_ssl else detach_k
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [mix, detach]).unsqueeze(-1)
            # do mixup shuffle for detach_
            detach_, _, _ = batch_shuffle_ddp(detach_, idx_shuffle=idx_shuffle_mix)
            l_pos_ = torch.einsum('nc,nc->n', [mix, detach_]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [mix, self.queue.clone().detach()])
            loss_mix = lam * self.head(l_pos, l_neg)["loss"] + (1 - lam) * self.head(l_pos_, l_neg)["loss"]
            
            losses["loss"] = loss_moco['loss'] * self.weight_ssl + loss_mix * self.weight_mix
        else:
            raise NotImplementedError
        
        # update K
        self._dequeue_and_enqueue(detach_k)
        return losses

    def forward_moco(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)[-1]  # queries: NxC
        q = nn.functional.normalize(self.neck_q([q])[0], dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            im_cat = torch.cat([im_q, im_k])
            im_cat, _, idx_unshuffle = batch_shuffle_ddp(im_cat)

            # mixup
            detach = self.encoder_k(im_cat)[-1]
            detach = self.neck_k([detach])[0]  # keys: NxC
            detach = nn.functional.normalize(detach, dim=1)
            # undo shuffle
            detach = batch_unshuffle_ddp(detach, idx_unshuffle)
            detach_q, detach_k = detach[:im_q.size(0)], detach[im_q.size(0):]

        # MoCo compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, detach_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        loss_moco = self.head(l_pos, l_neg)
        return loss_moco, detach_k, detach_q

    def forward_mix(self, detach_q, detach_k, im_mixed, encoder_str, neck_str, lam, idx_unshuffle_mix):
        # shuffle for making use of BN
        im_mixed, _, idx_unshuffle = grad_batch_shuffle_ddp(im_mixed)
        encoder_func = getattr(self, encoder_str)
        im_mixed = encoder_func(im_mixed)[-1]
        neck_func = getattr(self, neck_str)
        mix = nn.functional.normalize(neck_func([im_mixed])[0], dim=1)
        # undo forward shuffle
        mix = grad_batch_unshuffle_ddp(mix, idx_unshuffle)

        # method notice: assuming cross_view_gen, i.e., im_mixed = im_q + im_k (mixup shuffle)
        # cross_view_ssl is True: we caculate the lambda postive pair of mix and detach_k, and the
        #   1-lambda positive pair of mix and detach_q (mixup shuffled).
        detach  = detach_q if not self.cross_view_ssl else detach_k
        detach_ = detach_q if self.cross_view_gen == self.cross_view_ssl else detach_k
        
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [mix, detach]).unsqueeze(-1)
        # undo mixup shuffle
        mix = grad_batch_unshuffle_ddp(mix, idx_unshuffle_mix)
        l_pos_ = torch.einsum('nc,nc->n', [mix, detach_]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [mix, self.queue.clone().detach()])
        loss_mix = lam * self.head(l_pos, l_neg)["loss"] + (1 - lam) * self.head(l_pos_, l_neg)["loss"]
        return dict(loss=loss_mix)
    
    @torch.no_grad()
    def plot_mix(self, im_mixed, im, im_, lam, name=""):
        invTrans = transforms.Compose([
            transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1/0.2023, 1/0.1994, 1/0.201]),
            transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[ 1., 1., 1. ])])
        img = torch.cat((im[:4], im_[:4], im_mixed[:4]), dim=0)
        img_grid = torchvision.utils.make_grid(img, nrow=4, pad_value=0)
        img = np.transpose(invTrans(img_grid).detach().cpu().numpy(), (1, 2, 0))
        fig = plt.figure()
        plt.imshow(img)
        plt.title('lambda {}: {}'.format(name, lam))
        if not os.path.exists(self.save_name):
            plt.savefig(self.save_name)
        plt.close()
