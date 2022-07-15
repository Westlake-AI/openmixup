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
from ..utils import concat_all_gather, GatherLayer, Smoothing


@MODELS.register_module
class MoCoSAMix(BaseModel):
    """MOCO + SAMix

    *** this version is current sota, but has infomation leaky in mix_idx_shuffle ***

    Official implementation of
        "Boosting Discriminative Visual Representation Learning with Scenario-Agnostic
            Mixup (https://arxiv.org/pdf/2111.15454.pdf)"

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        head_clst (dict): Config dict for clustering module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        pretrained_k (str, optional): Path to pre-trained weights for en_k. Default: None.
        queue_len (int): Number of negative keys maintained in the queue.
            Default: 65536.
        feat_dim (int): Dimension of compact feature vectors. Default: 128.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
        alpha (int): Beta distribution '$\beta(\alpha, \alpha)$'.
        mask_layer (int): Number of the feature layer indix in the backbone.
        mask_adjust (float): Probrobality (in [0, 1]) of adjusting the mask (q) in terms
            of lambda (q), which only affect the backbone training. "auto" denotes adjusting
            by lambda, "auto_label" denotes adjusting in terms of (pseudo) labels and lambda.
            Default: 'auto' (or 0.).
        mean_margin (float): Margin of lambda (in (0,1)) to 'auto' adjust mask (q) with lam.
            Default: 0.1.
        mask_smooth (float): Smoothing the mixup mask with Gaussian Kernel for backbone. The
            kernel bandwith is chosen from [0.5, mask_smooth>1].
            Default: False. (or 0.)
        pre_one_loss (float): Loss weight for the pre-MixBlock head as onehot classification.
            Default: 0. (requires a pre_head in MixBlock)
        pre_mix_loss (float): Loss weight for the pre-MixBlock head as mixup classification.
            Default: 0. (requires a pre_head in MixBlock)
        lam_margin (int): Margin of lambda (in (0,1)) to stop using AutoMix to train backbone
            when lam is small. If lam_margin < lam < 1-lam_margin: AutoMix; else: mixup.
            Default: -1 (or 0).
        main_mb_loss (str): The main loss for training MixBlock, ["infoNCE", "BCE", "CE"].
            'BCE' denotes 'lam * BCE(mix*q, mix*k) + (1-lam) * BCE(mix*q, mix*k)'.
            'CE' denotes parametric classification loss in the clustering mode.
            Default: "infoNCE" (using the infoNCE loss).
        auxi_mb_loss (bool or float): The auxiliary loss for training Mix Block including
            ["BCE", "infoNCE"]. This loss needs latent space feature augmentation, then
            caculate mixup loss between mixture of the pos pair [q1,q2] and [k1,k2],
            e.g., BCE(mix*MIX(q1,q2,...), mix*MIX(k1,k2,...)) * lam + \
                BCE(mix*MIX(k1,k2,...), mix*MIX(q1,q2,...)) * (1-lam) for binary cls.
            Default: None or "None".
        feat_pos_extend (str): The latent space pos features extending methods using mixing,
            including ["interpolation", "expolation", "both", None]
        mix_shuffle_no_repeat (bool): [MixBlock] Whether to use 'no_repeat' mode to generate mixup
            dist shuffle idx when training Mix Block.
            Default: False.
        clustering (dict): Config dict that specifies the clustering algorithm, e.g.,
                clustering=dict(type="ODC", memory_bank=dict(type='ODCMemory', ...))
                clustering=dict(type="DeepCluster",)
            Default: None (no clustering).
        loss_weights (dict): Dict of the used the loss names and loss weights.
            Default: dict(
                decent_weight=["weight_mb_auxi"], accent_weight=[],
                weight_bb_mix=1, weight_bb_ssl=1, weight_mb_main=1, weight_mb_auxi=1,
                weight_mb_pre=1, weight_mb_mask=1)
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 head_clst=None,
                 mix_block=None,
                 pretrained=None,
                 pretrained_k=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 alpha=2,
                 mask_layer=2,
                 mask_adjust="auto",
                 mean_margin=0.1,
                 mask_smooth=0.,
                 pre_one_loss=0.,
                 pre_mix_loss=0.,
                 lam_margin=-1,
                 main_mb_loss="infoNCE",
                 auxi_mb_loss=None,
                 feat_pos_extend="both",
                 mix_shuffle_no_repeat=False,
                 clustering=dict(type="DeepCluster", nonlinear_neck=False),
                 loss_weights=dict(
                    decent_weight=["weight_mb_auxi", "weight_mb_mask"], accent_weight=[],
                    weight_bb_mix=1, weight_bb_ssl=1, weight_mb_main=1, weight_mb_auxi=0.5,
                    weight_mb_pre=1, weight_mb_mask=0.1),
                 save=False,
                 save_name="MixedSamples",
                 debug=False,
                 init_cfg=None,
                 **kwargs):
        super(MoCoSAMix, self).__init__(init_cfg, **kwargs)
        # build basic networks
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.encoder_q = builder.build_backbone(backbone)
        self.encoder_k = builder.build_backbone(backbone)
        self.neck_q = builder.build_neck(neck)
        self.neck_k = builder.build_neck(neck)
        self.backbone = self.encoder_q  # for feature extract
        self.head = builder.build_head(head)

        self.mix_block = mix_block
        if mix_block is not None:
            self.mix_block = builder.build_head(mix_block)
        
        self.head_clst = head_clst
        if head_clst is not None:
            assert isinstance(clustering, dict) and \
                clustering['type'] in ["ODC", "DeepCluster"]
            print_log('clustering -- {}'.format(clustering['type']), logger='root')

            # cluster head: cls
            self.head_clst = builder.build_head(head_clst)
            self.head_clst_off = builder.build_head(head_clst)
            
            # DeepCluster or ODC reweight
            self.num_classes = head_clst.num_classes
            self.loss_weight = torch.ones((self.num_classes, ), dtype=torch.float32).cuda()
            self.loss_weight /= self.loss_weight.sum()
            
            # memory bank for ODC
            if clustering['type'] == "ODC":
                self.memory_bank = builder.build_memory(clustering['memory_bank'])
                self.neck = self.neck_k
            elif clustering['type'] == "DeepCluster":
                if clustering.get('nonlinear_neck', False):
                    self.neck = self.neck_k
                else:
                    self.neck = builder.build_neck(dict(type='AvgPoolNeck'))  # for feature extract
            self.clustering = clustering['type']
        else:
            self.clustering = None

        # basic args
        self.queue_len = int(queue_len)
        self.momentum = float(momentum)
        self.alpha = float(alpha)
        self.mask_layer = int(mask_layer)
        self.mask_adjust = mask_adjust
        self.mean_margin = float(mean_margin)
        self.mask_smooth = float(mask_smooth)
        self.pre_one_loss = float(pre_one_loss) if float(pre_one_loss) > 0 else 0
        self.pre_mix_loss = float(pre_mix_loss) if float(pre_mix_loss) > 0 else 0
        self.lam_margin = float(lam_margin)
        self.auxi_mb_loss = str(auxi_mb_loss)
        self.feat_pos_extend = str(feat_pos_extend)
        self.main_mb_loss = str(main_mb_loss)
        self.mix_shuffle_no_repeat = bool(mix_shuffle_no_repeat)
        self.save = bool(save)
        self.save_name = str(save_name)
        self.debug = bool(debug)
        assert lam_margin < 1 and mask_layer <= 4
        # adjust mask
        if isinstance(mask_adjust, str):
            assert mask_adjust in ["auto", "auto_label"]
            if mask_adjust == "auto_label":
                assert self.clustering is not None
        if isinstance(mask_adjust, float):
            # assert mask_adjust in [0, 1]
            self.mask_adjust = None if (mask_adjust < 0 or mask_adjust > 1) else float(mask_adjust)
        # FFN before mixblock
        if self.pre_one_loss > 0 or self.pre_mix_loss > 0:
            assert self.mix_block.pre_neck is not None and \
                self.mix_block.pre_head is not None
        # mixup mask smoothing
        if self.mask_smooth < 2:
            self.mask_smooth = False
        self.smoother = Smoothing() if self.mask_smooth else None
        # main loss for mixblock
        assert self.main_mb_loss in ["infoNCE", "BCE", "CE"]
        # auxiliary loss for mixblock
        if self.auxi_mb_loss not in ["infoNCE", "BCE"]:
            self.auxi_mb_loss = None
        if self.feat_pos_extend not in ["interpolation", "expolation", "both"]:
            self.feat_pos_extend = None
        # loss weights for backbone (bb) and MixBlock (mb)
        self.loss_weights = loss_weights
        for key in loss_weights.keys():
            if not isinstance(loss_weights[key], list):
                self.loss_weights[key] = \
                    float(loss_weights[key]) if float(loss_weights[key]) > 0 else 0
        self.weight_bb_mix = loss_weights.get("weight_bb_mix", 1.)
        self.weight_bb_ssl = loss_weights.get("weight_bb_ssl", 1.)
        self.weight_mb_main = loss_weights.get("weight_mb_main", 1.)
        self.weight_mb_auxi = loss_weights.get("weight_mb_auxi", 1.)
        self.weight_mb_mask = loss_weights.get("weight_mb_mask", 1.)
        self.weight_mb_pre = loss_weights.get("weight_mb_pre", 1.)
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine

        # create the MoCo queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(MoCoSAMix, self).init_weights()

        if pretrained is not None:
            print_log('load encoder_q from: {}'.format(pretrained), logger='root')
        self.encoder_q.init_weights(pretrained=pretrained)
        if pretrained_k is not None:
            print_log('load encoder_k from: {}'.format(pretrained), logger='root')
        self.encoder_k.init_weights(pretrained=pretrained_k)

        self.neck_q.init_weights(init_linear='kaiming')
        if self.mix_block is not None:
            self.mix_block.init_weights(init_linear='normal')
        # stop grad for head_clst_off
        if self.head_clst is not None:
            self.head_clst.init_weights(init_linear='normal')
            for param_q, param_k in zip(self.head_clst.parameters(),
                                        self.head_clst_off.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
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
            if self.momentum <= 0.:
                param_k.data = param_q.data
            elif self.momentum < 1.:
                param_k.data = param_k.data * self.momentum + \
                            param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.neck_q.parameters(),
                                    self.neck_k.parameters()):
            if self.momentum <= 0.:
                param_k.data = param_q.data
            elif self.momentum < 1.:
                param_k.data = param_k.data * self.momentum + \
                            param_q.data * (1. - self.momentum)
        
        # update cluster head
        if self.head_clst is not None and self.head_clst_off is not None:
            for param_q, param_k in zip(self.head_clst.parameters(),
                                        self.head_clst_off.parameters()):
                param_k.data.copy_(param_q.data)

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

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle, idx_shuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def _grad_batch_shuffle_ddp(self, x, idx_shuffle=None, no_repeat=False):
        """Batch shuffle, for making use of BatchNorm. (SimCLR GatherLayer)

            *** Only support DistributedDataParallel (DDP) model. ***
        Args:
            idx_shuffle: Given shuffle index if not None.
            no_repeat: The idx_shuffle does not have any repeat index as
                the original indice [i for i in range(N)]. It's used in
                mixup methods (self-supervisedion).
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = torch.cat(GatherLayer.apply(x), dim=0)  # with grad
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        if idx_shuffle is None:
            # generate shuffle idx
            idx_shuffle = torch.randperm(batch_size_all).cuda()
            # each idx should not be the same as the original
            if bool(no_repeat) == True:
                idx_original = torch.tensor([i for i in range(batch_size_all)]).cuda()
                idx_repeat = False
                for i in range(30):  # try 30 times
                    if (idx_original == idx_shuffle).any() == True:
                        idx_repeat = True
                        idx_shuffle = torch.randperm(batch_size_all).cuda()
                    else:
                        idx_repeat = False
                        break
                # hit: prob < 2.4e-6
                if idx_repeat == True:
                    fail_to_shuffle = True
                    idx_shuffle = idx_original.clone()
                    for i in range(3):
                        # way 1: repeat prob < 1.5e-5
                        rand_ = torch.randperm(batch_size_all).cuda()
                        idx_parition = rand_ > torch.median(rand_)
                        idx_part_0 = idx_original[idx_parition == True]
                        idx_part_1 = idx_original[idx_parition != True]
                        if idx_part_0.shape[0] == idx_part_1.shape[0]:
                            idx_shuffle[idx_parition == True] = idx_part_1
                            idx_shuffle[idx_parition != True] = idx_part_0
                            if (idx_original == idx_shuffle).any() != True:  # no repeat
                                fail_to_shuffle = False
                                break
                    # fail prob -> 0
                    if fail_to_shuffle:
                        # way 2: repeat prob = 0, but too simple!
                        idx_shift = np.random.randint(1, batch_size_all-1)
                        idx_shuffle = torch.tensor(  # shift the original idx
                            [(i+idx_shift) % batch_size_all for i in range(batch_size_all)]).cuda()
        else:
            assert idx_shuffle.size(0) == batch_size_all, \
                "idx_shuffle={}, batchsize={}".format(idx_shuffle.size(0), batch_size_all)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle, idx_shuffle

    def _grad_batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle. (SimCLR GatherLayer)

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = torch.cat(GatherLayer.apply(x), dim=0)  # with grad
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def _update_loss_weights(self):
        """ update loss weights according to the cos_annealing scalar """
        # cos annealing decent
        for attr in self.loss_weights["decent_weight"]:
            setattr(self, attr, self.loss_weights[attr] * self.cos_annealing)  # from 1 to 0
        # cos annealing accent
        for attr in self.loss_weights["accent_weight"]:
            setattr(self, attr, self.loss_weights[attr] * (1-self.cos_annealing))  # from 0 to 1

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list) and len(img) >= 2
        im_q = img[0].contiguous()
        im_k = img[1].contiguous()
        # update loss weights
        self._update_loss_weights()

        # sample two lambdas from beta distribution
        lam = np.random.beta(self.alpha, self.alpha, 2) # 0 for q; 1 for k

        ########################################## MOCO ############################################
        loss_moco, f_q, f_k, detach_q, detach_k = self.forward_moco(im_q, im_k)

        ######################################## AutoMix ###########################################
        # im_mixed_bb, im_mixed_mb, idx_shuffle_bb, idx_shuffle_mb, loss_mask = 
        results = self.pixel_mix(im_q, im_k, lam, f_q, f_k, **kwargs)

        # whether to using clustering to train MixBlock
        cluster_mode = (self.main_mb_loss == "CE") and \
            (self.head_clst is not None) and (kwargs.get('pseudo_label', None) is not None)

        # loss for backbone q
        loss_bb = self.forward_mix(
            encoder_str="encoder_q", neck_str="neck_q", detach_q=detach_k, detach_k=detach_q,
            im_mixed=results["im_mixed_bb"], lam=lam[0], idx_shuffle_k=results["idx_shuffle_bb"],
            binary_cls=False, mixblock_mode=False)
        # loss for mixblock k
        if not cluster_mode:
            # loss for mix block, try 2 cls loss or not
            loss_mb = self.forward_mix(
                encoder_str="encoder_k", neck_str="neck_k", detach_q=detach_k, detach_k=detach_q,
                im_mixed=results["im_mixed_mb"], lam=lam[1], idx_shuffle_k=results["idx_shuffle_mb"],
                binary_cls=bool(self.main_mb_loss=="BCE"), mixblock_mode=True,
                mb_x_lam=results["x_lam"], mb_x_lam_=results["x_lam_"])
        else:
            loss_mb = self.forward_mix_cluster(
                encoder_str="encoder_k", neck_str="neck_k", detach_q=detach_k, detach_k=detach_q,
                im_raw=im_q, im_mixed=results["im_mixed_mb"], lam=lam[1], idx_shuffle_k=results["idx_shuffle_mb"],
                pseudo_label=kwargs['pseudo_label'], idx=kwargs['idx'],)
        
        # loss summary
        losses = dict()
        losses["loss"] = loss_mb['loss'] + \
            self.weight_bb_ssl * loss_moco['loss'] + self.weight_bb_mix * loss_bb['loss']
        # mixblock
        if results["mask_loss"] is not None and self.weight_mb_mask > 0:
            losses["loss"] += results["mask_loss"] * self.weight_mb_mask
        if loss_mb["pre_one_loss"] is not None and self.pre_one_loss > 0:
            losses["loss"] += loss_mb["pre_one_loss"] * self.weight_mb_pre
        if loss_mb["pre_mix_loss"] is not None and self.pre_mix_loss > 0:
            losses["loss"] += loss_mb["pre_mix_loss"] * self.weight_mb_pre
        
        self._dequeue_and_enqueue(detach_k)
        return losses

    def forward_moco(self, im_q, im_k):
        """ original MoCo forward """
        # compute query features
        q = self.encoder_q(im_q)[-1]  # queries: NxC
        q = nn.functional.normalize(self.neck_q([q])[0], dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_cat = torch.cat([im_q, im_k])
            im_cat, idx_unshuffle, _ = self._batch_shuffle_ddp(im_cat)

            detach = self.encoder_k(im_cat)
            if len(detach) == 2:
                f, detach = tuple(detach)  # NxCxHxW, NxCxhxw
            elif len(detach) == 1:
                detach = detach[0]  # keys: NxCxhxw
                f = detach.clone()
            detach = nn.functional.normalize(self.neck_k([detach])[0], dim=1)

            # undo shuffle
            f = self._batch_unshuffle_ddp(f, idx_unshuffle)
            detach = self._batch_unshuffle_ddp(detach, idx_unshuffle)
                
            f_q, f_k = f[:im_q.size(0)], f[im_q.size(0):]
            detach_q, detach_k = detach[:im_q.size(0)], detach[im_q.size(0):]

        # compute logits, Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, detach_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        loss_moco = self.head(l_pos, l_neg)
        return loss_moco, f_q, f_k, detach_q, detach_k

    def forward_mix(self,
                    encoder_str, neck_str, detach_q, detach_k,
                    im_mixed, lam, idx_shuffle_k,
                    binary_cls=False, mixblock_mode=False,
                    mb_x_lam=None, mb_x_lam_=None):
        """ mixup forward for both the backbone and mixblock """
        # shuffle for making use of BN
        im_mixed, idx_unshuffle, _ = self._grad_batch_shuffle_ddp(im_mixed)
        encoder_func = getattr(self, encoder_str)
        im_mixed = encoder_func(im_mixed)
        neck_func = getattr(self, neck_str)
        mix = nn.functional.normalize(neck_func([im_mixed[-1]])[0], dim=1)
        # undo forward shuffle
        mix = self._grad_batch_unshuffle_ddp(mix, idx_unshuffle)

        # pos logits: Nx1, lam
        l_pos = torch.einsum('nc,nc->n', [mix, detach_q]).unsqueeze(-1)  # mix and q
        # undo mixup shuffle
        detach_k_, _, _ = self._grad_batch_shuffle_ddp(detach_k, idx_shuffle_k)  # mixup shuffle
        # pos logits: Nx1, 1-lam
        l_pos_ = torch.einsum('nc,nc->n', [mix, detach_k_]).unsqueeze(-1)  # mix and k
        # negative logits: NxK
        l_neg = None
        if binary_cls == False:
            l_neg = torch.einsum('nc,ck->nk', [mix, self.queue.clone().detach()])
        
        # extending pos pairs for the auxiliary MB loss
        if self.auxi_mb_loss is not None and mixblock_mode == True:
            if self.feat_pos_extend is not None:
                if self.feat_pos_extend == "both":
                    prob_ex = bool(np.random.rand(1) > 0.5)  # 1/2 expolation, 1/2 interpolation
                else:
                    prob_ex = self.feat_pos_extend == "expolation"
                mix_pos_q, mix_pos_k = self.pos_feat_extension(
                    detach_q, detach_k, mix_distribution="beta", pos_alpha=2, mix_expolation=prob_ex)
            else:
                mix_pos_q, mix_pos_k = detach_q, detach_k
            m_pos  = torch.einsum('nc,nc->n', [mix, mix_pos_q]).unsqueeze(-1)  # Nx1
            mix_pos_k, _, _ = self._grad_batch_shuffle_ddp(mix_pos_k, idx_shuffle_k)  # mixup shuffle
            m_pos_ = torch.einsum('nc,nc->n', [mix, mix_pos_k]).unsqueeze(-1)  # Nx1
            if self.auxi_mb_loss == "infoNCE" and binary_cls:
                l_neg = torch.einsum('nc,ck->nk', [mix, self.queue.clone().detach()])
        
        # main MixBlock loss: normal infoNCE loss
        if binary_cls == False:
            loss_mix = self.weight_mb_main * \
                (lam * self.head(l_pos, l_neg)["loss"] + (1 - lam) * self.head(l_pos_, l_neg)["loss"])
        # BCE loss {lam, 1-lam} for mixblock
        else:
            loss_mix = self.weight_mb_main * \
                (lam * self.head(l_pos, l_pos_)["loss"] + (1 - lam) * self.head(l_pos_, l_pos)["loss"])
        
        # auxiliary MixBlock loss
        if mixblock_mode == True:
            # augmented pos pair loss using BCE
            if self.auxi_mb_loss == "BCE":
                loss_mix += self.weight_mb_auxi * \
                    (self.head(m_pos, m_pos_)["loss"] * lam + self.head(m_pos_, m_pos)["loss"] * (1 - lam))
            # augmented pos pair loss for InfoNCE
            elif self.auxi_mb_loss == "infoNCE":
                loss_mix += self.weight_mb_auxi * \
                    (self.head(m_pos, l_neg)["loss"] * lam + self.head(m_pos_, l_neg)["loss"] * (1 - lam))
        
        losses = dict(loss=loss_mix)
        
        # for pre mixblock loss
        losses["pre_mix_loss"] = None
        losses["pre_one_loss"] = None
        if mixblock_mode == True:
            # SimCLR infoNCE
            if self.pre_one_loss > 0 or self.pre_mix_loss > 0:
                mb_x_lam  = nn.functional.normalize(
                    self.mix_block.pre_neck([mb_x_lam])[0], dim=1)  # nxd
                mb_x_lam_ = nn.functional.normalize(
                    self.mix_block.pre_neck([mb_x_lam_])[0], dim=1)  # nxd
                losses["pre_one_loss"] = \
                    self.pre_one_loss * self.forward_mix_SimCLR(
                        torch.cat([mb_x_lam.unsqueeze(1), mb_x_lam_.unsqueeze(1)], dim=1))["loss"]
            # SimCLR mixup infoNCE
            if self.pre_mix_loss > 0:
                mb_mix = im_mixed[0]
                # mixblock pre FFN
                if self.mix_block.pre_attn is not None:
                    mb_mix = self.mix_block.pre_attn(mb_mix)  # non-local
                if self.mix_block.pre_conv is not None:
                    mb_mix = self.mix_block.pre_conv([mb_mix])  # neck
                mb_mix = nn.functional.normalize(
                    self.mix_block.pre_neck(mb_mix)[0], dim=1)  # nxd
                losses["pre_mix_loss"] = self.pre_mix_loss * \
                    (lam * self.forward_mix_SimCLR(
                        torch.cat([mb_mix.unsqueeze(1), mb_x_lam.unsqueeze(1)], dim=1))["loss"] + \
                    (1-lam) * self.forward_mix_SimCLR(
                        torch.cat([mb_mix.unsqueeze(1), mb_x_lam_.unsqueeze(1)], dim=1))["loss"])
        
        return losses

    def forward_mix_cluster(self,
                            encoder_str, neck_str, detach_q, detach_k,
                            im_raw, im_mixed, lam, idx_shuffle_k,
                            pseudo_label, idx):
        # detach_q, detach_k, encoder_str, neck_str, im_raw, pseudo_label, idx, im_mixed, lam, idx_shuffle_k):
        """ clustering forward for mixblock """
        # encoder
        encoder_func = getattr(self, encoder_str)
        mix = encoder_func(im_mixed)[-1]
        raw = encoder_func(im_raw)[-1]

        losses = dict()

        # cluster methods
        if self.clustering == "DeepCluster":
            # default neck (neck_k or GAP)
            mix = self.neck([mix])[0]
            raw = self.neck([raw])[0]

            # train clustering head
            mix_outs = self.head_clst([mix])
            raw_outs = self.head_clst([raw])
            # train mixblock
            mix_outs_mb = self.head_clst_off([mix])  # no grad head
        elif self.clustering == "ODC":
            # neck
            neck_func = getattr(self, neck_str)
            mix = neck_func([mix])[0]
            raw = neck_func([raw])[0]

            # train clustering head
            mix_outs = self.head_clst([mix])
            raw_outs = self.head_clst([raw])
            # train mixblock
            mix_outs_mb = self.head_clst_off([mix])  # no grad head

            # get labels from ODC memory_bank
            if self.memory_bank.label_bank.is_cuda:
                pseudo_label = self.memory_bank.label_bank[idx]
            else:
                pseudo_label = self.memory_bank.label_bank[idx.cpu()].cuda()

            # update samples memory
            change_ratio = self.memory_bank.update_samples_memory(idx, detach_k)
            losses['change_ratio'] = change_ratio
        else:
            raise NotImplementedError

        # mixup and onehot labels
        pseudo_label_, _, _ = self._grad_batch_shuffle_ddp(pseudo_label, idx_shuffle_k)
        y_mix = (pseudo_label, pseudo_label_, lam)  # mixup lam
        y_one = (pseudo_label, pseudo_label_, 1)  # onehot

        # mixup loss and onehot loss for clustering head
        loss_cluster = self.head_clst.loss(mix_outs, y_mix)['loss'] + self.head_clst.loss(raw_outs, y_one)['loss']
        # mixup loss for mixblock
        loss_mixblock = self.head_clst_off.loss(mix_outs_mb, y_mix)['loss']
        
        losses['loss'] = loss_cluster + loss_mixblock * self.weight_mb_main

        # for pre mixblock loss
        losses["pre_mix_loss"] = None
        losses["pre_one_loss"] = None

        # extending pos pairs for BCE mixup loss
        if self.auxi_mb_loss is not None:
            if self.feat_pos_extend is not None:
                if self.feat_pos_extend == "both":
                    prob_ex = bool(np.random.rand(1) > 0.5)  # 1/2 expolation, 1/2 interpolation
                else:
                    prob_ex = self.feat_pos_extend == "expolation"
                mix_pos_q, mix_pos_k = self.pos_feat_extension(
                    detach_q, detach_k, mix_distribution="beta", pos_alpha=2, mix_expolation=prob_ex)
            else:
                mix_pos_q, mix_pos_k = detach_q, detach_k
            m_pos  = torch.einsum('nc,nc->n', [mix, mix_pos_q]).unsqueeze(-1)  # Nx1
            mix_pos_k, _, _ = self._grad_batch_shuffle_ddp(mix_pos_k, idx_shuffle_k)  # mixup shuffle
            m_pos_ = torch.einsum('nc,nc->n', [mix, mix_pos_k]).unsqueeze(-1)  # Nx1
            # augmented pos pair loss using BCE (this head is the contrastive head)
            if self.auxi_mb_loss == "BCE":
                losses['loss'] += self.weight_mb_auxi * \
                    (self.head(m_pos, m_pos_)["loss"] * lam + self.head(m_pos_, m_pos)["loss"] * (1 - lam))
            # augmented pos pair loss for InfoNCE
            elif self.auxi_mb_loss == "infoNCE":
                l_neg = torch.einsum('nc,ck->nk', [mix, self.queue.clone().detach()])
                losses['loss'] += self.weight_mb_auxi * \
                    (self.head(m_pos, l_neg)["loss"] * lam + self.head(m_pos_, l_neg)["loss"] * (1 - lam))
        
        return losses

    @staticmethod
    def _create_buffer_SimCLR(N):
        """ create SimCLR mask of InfoNCE """
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def forward_mix_SimCLR(self, z):
        """ SimCLR mixup loss of the feature vector z """
        assert z.size(1) == 2, "Input should be [n, 2, d], got: {}".format(z.dim())
        # z is L2-normalized
        z = z.reshape(z.size(0) * 2, z.size(2))  # 2nxd
        # z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer_SimCLR(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        # using mixblock pre_head (diff from self.head)
        losses = self.mix_block.pre_head(positive, negative)
        return losses

    @torch.no_grad()
    def mix_mask_adjust(self, mask, lam, pseudo_label=None):
        """ adjust mask according to lam and (pseudo) labels
        
        Args:
            mask, lam: Input of mask (N, 2, H, W) and lam.
            mask_adjust (str or float): If "auto" mode, adjust mask based on mean_margin,
                elif float, adjust when mask_adjust > prob in [0,1]. Elif "auto_label" mode,
                using vanilla mixup for mixup samples from the same class and using 'auto' for
                the rest.
                Default: mask_adjust is None.
            mask_smooth (float): Smoothing the mixup mask by Gaussian Kernel.
        """
        # lam_margin for vanilla mixup
        if self.lam_margin >= lam or self.lam_margin >= (1 - lam):
            mask[:, 0, :, :] = lam
            mask[:, 1, :, :] = 1 - lam
        else:
            # apply Gaussian smoothing
            if self.mask_smooth:
                mask = mask[:, :1, :, :]  # lam
                sigma = np.random.uniform(0.5, self.mask_smooth)  # e.g., [0.5, 3]
                mask = self.smoother(mask, sigma=sigma)  # bandwidth = sigma * 5
                mask = torch.cat([mask, 1-mask], dim=1)
            
            if self.mask_adjust is None:
                return mask
            # adjust mask with pseudo labels
            if isinstance(self.mask_adjust, str):
                if self.mask_adjust == "auto_label":
                    assert isinstance(pseudo_label, list) and len(pseudo_label)==2
                    idx_same_class = pseudo_label[0] == pseudo_label[1]
                    mask[idx_same_class, 0, :, :] = lam
                    mask[idx_same_class, 1, :, :] = 1 - lam
            # adjust mask with lam
            epsilon = 1e-10
            _mask = mask[:, 0, :, :].squeeze()  # [N, H, W], _mask for lam
            _mask = _mask.clamp(min=epsilon, max=1-epsilon)
            _mean = _mask.mean(dim=[1, 2]).squeeze()  # [N, 1, 1] -> [N]
            idx_larg, idx_less = None, None
            if isinstance(self.mask_adjust, float):
                if self.mask_adjust > np.random.rand():  # [0,1)
                    idx_larg = _mean[:] > lam + epsilon  # index of mean > lam + eps
                    idx_less = _mean[:] < lam - epsilon  # index of mean < lam - eps
            else:  # 'auto' or 'auto_label'
                idx_larg = _mean[:] > lam + self.mean_margin  # index of mean > lam + margin
                idx_less = _mean[:] < lam - self.mean_margin  # index of mean < lam - margin
            
            # adjust mask according to idx_larg & idx_less
            if idx_larg is not None and idx_less is not None:
                # if mean > lam + m, idx_larg
                mask[idx_larg==True, 0, :, :] = \
                    _mask[idx_larg==True, :, :] * (lam / _mean[idx_larg==True].view(-1, 1, 1))
                mask[idx_larg==True, 1, :, :] = 1 - mask[idx_larg==True, 0, :, :]
                # elif mean < lam - m, idx_less
                mask[idx_less==True, 1, :, :] = \
                    (1 - _mask[idx_less==True, :, :]) * ((1 - lam) / (1 - _mean[idx_less==True].view(-1, 1, 1)))
                mask[idx_less==True, 0, :, :] = 1 - mask[idx_less==True, 1, :, :]
        
        return mask

    def pixel_mix(self, im_q, im_k, lam, f_q, f_k, pseudo_label=None, idx=None, **kwargs):
        """ pixel-wise input space mixup, v07.29

        Args:
            im_q, im_k (Tensor): Input of a batch of images, (N, C, H, W).
            lam (List): Input list of lambda (scalar).
            f_q, f_k (Tensor): The feature map of x, (N, C, H', W').
            pseudo_label (Tensor): The clustering pseudo labels for mask adjusting.

        Returns: dict includes
            im_mixed_bb, im_mixed_bb: Mixup samples for bb (training the backbone)
                and mb (training the mixblock).
            idx_unshuffle_bb, idx_unshuffle_mb: Mixup shuffle indice bb and mb.
            pre_one_loss: Output pre mixblock loss (like onehot cls).
            mask_loss: Output loss of mixup masks.
        """
        results = dict()
        # step 0: mask upsampling factor in ResNet
        if im_q.shape[3] > 64:  # normal version of resnet
            scale_factor = 2**(2 + self.mask_layer)
        else:  # CIFAR version
            scale_factor = 2**self.mask_layer
        
        # step 1: mixup bb, for backbone
        # mixup shuffle feature f_k
        f_k, _, idx_shuffle_bb = self._grad_batch_shuffle_ddp(f_k, no_repeat=self.mix_shuffle_no_repeat)
        results["idx_shuffle_bb"] = idx_shuffle_bb
        with torch.no_grad():
            mask_bb = self.mix_block(
                [f_q, f_k], lam[0], index=None, scale_factor=scale_factor)
            mask_bb = mask_bb["mask"].clone().detach()
        pseudo_label_ = None
        if pseudo_label is not None:  # pseudo labels for mask adjust
            if self.clustering == "DeepCluster":
                pseudo_label_, _, _ = self._grad_batch_shuffle_ddp(pseudo_label, idx_shuffle_bb)
                pseudo_label = [pseudo_label, pseudo_label_]
        if idx is not None and pseudo_label_ is None:  # pseudo labels for mask adjust
            # get labels from ODC memory_bank
            assert self.clustering == "ODC"
            if self.memory_bank.label_bank.is_cuda:
                pseudo_label = self.memory_bank.label_bank[idx]
            else:
                pseudo_label = self.memory_bank.label_bank[idx.cpu()].cuda()
            pseudo_label_, _, _ = self._grad_batch_shuffle_ddp(pseudo_label, idx_shuffle_bb)
            pseudo_label = [pseudo_label, pseudo_label_]
        
        # adjust mask_bb with lam[0]
        mask_bb = self.mix_mask_adjust(mask_bb, lam[0], pseudo_label)
        
        # step 2: mixup mb, for mixblock
        # mixup shuffle feature f_k
        f_k, _, idx_shuffle_mb = self._grad_batch_shuffle_ddp(f_k, no_repeat=self.mix_shuffle_no_repeat)
        results["idx_shuffle_mb"] = idx_shuffle_mb
        if not self.debug:
            mask_mb = self.mix_block(
                [f_q, f_k], lam[1], index=None, scale_factor=scale_factor)
            debug_plot = None
        else:
            mask_mb = self.mix_block(
                [f_q, f_k], lam[1], index=None, scale_factor=scale_factor, debug=self.debug)
            debug_plot = mask_mb["debug_plot"]
        
        # pre mixblock loss
        results["x_lam"], results["x_lam_"] = None, None
        if self.pre_one_loss > 0 or self.pre_mix_loss > 0:
            results["x_lam"], results["x_lam_"] = mask_mb["x_lam"], mask_mb["x_lam_"]
        mask_mb = mask_mb["mask"]
        # loss of mixup mask
        results["mask_loss"] = None
        if self.weight_mb_mask > 0:
            results["mask_loss"] = self.mix_block.mask_loss(mask_mb, lam[1])["loss"]
        
        # step 3: generate mixup img
        # img mixup bb for backbone
        im_k_shuffle, _, _ = self._grad_batch_shuffle_ddp(im_k, idx_shuffle_bb)
        results["im_mixed_bb"] = \
            im_q * mask_bb[:, 0, :, :].unsqueeze(1) + im_k_shuffle * mask_bb[:, 1, :, :].unsqueeze(1)
        # save img bb
        if self.save and self.mask_adjust is not None:
            self.plot_mix(results["im_mixed_bb"], im_q, im_k_shuffle, lam[0], debug_plot, "backbone")
        
        # img mixup mb for mixblock
        im_k_shuffle, _, _ = self._grad_batch_shuffle_ddp(im_k, idx_shuffle_mb)
        results["im_mixed_mb"] = \
            im_q * mask_mb[:, 0, :, :].unsqueeze(1) + im_k_shuffle * mask_mb[:, 1, :, :].unsqueeze(1)
        # save img mb
        if self.save and self.mask_adjust is None:
            self.plot_mix(results["im_mixed_mb"], im_q, im_k_shuffle, lam[1], debug_plot, "mixblock")
        
        return results
    
    @torch.no_grad()
    def plot_mix(self, im_mixed, im_q, im_k, lam, debug_plot=None, name="k"):
        """ visualize mixup results, supporting 'debug' mode """
        invTrans = transforms.Compose([
            transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1/0.2023, 1/0.1994, 1/0.201]),
            transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[ 1., 1., 1. ])])
        # plot mixup results
        img = torch.cat((im_q[:4], im_k[:4], im_mixed[:4]), dim=0)
        img_grid = torchvision.utils.make_grid(img, nrow=4, pad_value=0)
        img = np.transpose(invTrans(img_grid).detach().cpu().numpy(), (1, 2, 0))
        fig = plt.figure()
        plt.imshow(img)
        plt.title('lambda {}={}'.format(name, lam))
        assert self.save_name.find(".png") != -1
        if not os.path.exists(self.save_name):
            plt.savefig(self.save_name)
        # debug: plot intermediate results
        if self.debug:
            assert isinstance(debug_plot, dict)
            for key,value in debug_plot.items():
                n, h, w = value.size()
                img = value[:4].view(h, 4 * w).detach().cpu().numpy()
                fig = plt.figure()
                plt.imshow(img)
                # plt.title('debug {}, lambda k={}'.format(str(key), lam))
                _debug_path = self.save_name.split(".png")[0] + "_{}.png".format(str(key))
                if not os.path.exists(_debug_path):
                    plt.savefig(_debug_path, bbox_inches='tight')
        plt.close()
    
    def pos_feat_extension(self, pos_q, pos_k, postmix_norm=True,
                    mix_expolation=False, mix_distribution='beta', pos_alpha=2):
        """ Extending positive pairs based on latent space mixing 
        
        Reference: "Improving Contrastive Learning by Visualizing Feature
            Transformation (https://arxiv.org/pdf/2108.02982.pdf)".
        Part of the code is based on
        "https://github.com/DTennant/CL-Visualizing-Feature-Transformation/blob/master/memory/mem_moco.py"

        Args:
            pos_q, pos_k (tensor): Projection features of a positive pair.
            postmix_norm (bool): Whether to use L2 norm after mixing.
            mix_expolation (bool): Whether to adopt expolation pos mixup (else using
                interpolation as the normal mixup).
            mix_distribution (str): Generating the mixing mask (tensor) based on
                ['beta', 'uniform'] distribution.
            pos_alpha (float): The alpha (Beta distribution) for pos pairs mixing.
        """
        assert pos_q.dim() == 2 and pos_k.dim() == 2
        mask_shape = pos_q.shape
        # mixup with a tensor mask
        if mix_distribution == 'uniform':
            mask = torch.rand(size=mask_shape).cuda()
        elif mix_distribution == 'beta':
            _Beta = torch.distributions.Beta(pos_alpha, pos_alpha)
            mask = _Beta.sample(mask_shape).cuda()
        else:
            mask = torch.ones(mask_shape).cuda() / 2
        # interpolation or expolation
        if mix_expolation:
            mask += 1
        
        mix_q = mask * pos_q + (1 - mask) * pos_k
        mix_k = mask * pos_k + (1 - mask) * pos_q
        # post mixing L2 norm to the hyper-sphere
        if postmix_norm:
            mix_q = nn.functional.normalize(mix_q, dim=1)
            mix_k = nn.functional.normalize(mix_k, dim=1)
        return mix_q, mix_k

    def set_reweight(self, labels=None, reweight_pow=0.5):
        """Loss re-weighting. DeepCluster or ODC

        Re-weighting the loss according to the number of samples in each class.

        Args:
            labels (numpy.ndarray): Label assignments.
            reweight_pow (float): The power of re-weighting. Default: 0.5.
        """
        if self.clustering == "DeepCluster":
            hist = np.bincount(
                labels, minlength=self.num_classes).astype(np.float32)
            inv_hist = (1. / (hist + 1e-10))**reweight_pow
            weight = inv_hist / inv_hist.sum()
        elif self.clustering == "ODC":
            if labels is None:
                if self.memory_bank.label_bank.is_cuda:
                    labels = self.memory_bank.label_bank.cpu().numpy()
                else:
                    labels = self.memory_bank.label_bank.numpy()
            hist = np.bincount(
                labels, minlength=self.num_classes).astype(np.float32)
            inv_hist = (1. / (hist + 1e-5))**reweight_pow
            weight = inv_hist / inv_hist.sum()
        else:
            raise NotImplementedError
        
        self.loss_weight.copy_(torch.from_numpy(weight))
        # cluster head
        self.head_clst.criterion.class_weight = self.loss_weight
        self.head_clst_off.criterion.class_weight = self.loss_weight

    def forward_test(self, img, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        # deep cluster head test
        x = self.backbone(img)  # tuple
        assert self.head_clst is not None
        outs = self.head_clst(x)
        keys = [f'head{i}' for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))
