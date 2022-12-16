from matplotlib.pyplot import axis
import torch
import torch.nn as nn
from mmcv.utils.parrots_wrapper import _BatchNorm
import numpy as np

from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS
from ..augments import cutmix, fmix, gridmix, mixup, resizemix, saliencymix, smoothmix


@MODELS.register_module
class DMixTuning(BaseModel):
    """
    Implementation of DMix-Tuning (using Decoupled Mixup)
        based on Self-Tuning (https://arxiv.org/pdf/2102.12903.pdf) and mixup methods.
    
    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Default: None.
        head_one: Config dict for classification head.
        head_mix: Config dict for mixup classification head.
        queue_size (int): Number of class-specific keys maintained in the queue
            (for the momentum queue). Default: 32.
        proj_dim (int): Dimension of the projector neck (for the momentum queue).
            Default: 128.
        class_num: Total class number of the dataset.
        pretrained: loading from pre-trained model or not (default: True)
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
        temperature (float): The temperature hyper-parameter that controls the
            concentration level of the distribution. Default: 0.07.
        alpha (float): To sample Beta distribution in MixUp methods.
        mix_mode (str): Basice mixUp methods in input space. Default: "mixup".
        mix_args (dict): Args for manifoldmix, resizeMix, fmix mode.
        label_rescale (str): Mixup label rescale based on lam, including ['labeled',
            'unlabeled', 'both', 'none']. Default: 'labeled'.
        lam_bias (str): Whether to use biased lam, i.e., lam>0.5, for ['labeled',
            'unlabeled', 'rand']. Default: 'labeled'.
        freeze_bn_unlabeled (bool): Whether to freeze bn when training unlabeled data.
        loss_weights (dict): Weights of each loss iterms, 'weight_pgc' for the (L/UL)
            PGC loss, 'weight_one' for the (L) one-hot loss, 'weight_mix_ll' for the
            mixup loss between two labeled samples, and 'weight_mix_lu' for the mixup
            loss between a labeled and unlabeled samples. Default to 1.
    """
    
    def __init__(self,
                 backbone,
                 neck=None,
                 head_one=None,
                 head_mix=None,
                 queue_size=32,
                 proj_dim=128,
                 class_num=200, 
                 momentum=0.999, 
                 temperature=0.07,
                 alpha=1.0,
                 mix_mode="mixup",
                 mix_args=dict(
                    manifoldmix=dict(layer=(0, 3)),
                    resizemix=dict(scope=(0.1, 0.8), use_alpha=False),
                    fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False)
                 ),
                 label_rescale='labeled',
                 lam_bias='labeled',
                 freeze_bn_unlabeled=False,
                 loss_weights=dict(
                    decent_weight=[],
                    accent_weight=['weight_mix_lu'],
                    weight_pgc=1, weight_one=1, weight_mix_ll=1, weight_mix_lu=1),
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(DMixTuning, self).__init__(init_cfg, **kwargs)
        # network settings
        self.encoder_q = builder.build_backbone(backbone)
        self.encoder_k = builder.build_backbone(backbone)
        self.projector_q = builder.build_neck(neck)
        self.projector_k = builder.build_neck(neck)
        self.backbone = self.encoder_q
        self.head_one = None
        self.head_mix = None
        if head_one is not None:
            self.head_one = builder.build_head(head_one)
            self.head = self.head_one
        if head_mix is not None:
            self.head_mix = builder.build_head(head_mix)
        self.init_weights(pretrained=pretrained)
        
        # Self-Tuning args
        self.queue_size = queue_size
        self.momentum = momentum
        self.class_num = class_num
        self.pretrained = pretrained
        self.temperature = temperature
        self.KL = nn.KLDivLoss(reduction='batchmean')

        # mixup args
        assert mix_mode in [
            "mixup", "manifoldmix", "cutmix", "saliencymix", "resizemix", "fmix"]
        if mix_mode in ["manifoldmix"]:
            assert 0 == min(mix_args[mix_mode]["layer"]) and max(mix_args[mix_mode]["layer"]) < 4
        if mix_mode == "resizemix":
            assert 0 <= min(mix_args[mix_mode]["scope"]) and max(mix_args[mix_mode]["scope"]) <= 1
        self.mix_mode = mix_mode
        self.alpha = alpha
        self.mix_args = mix_args
        assert label_rescale in ['labeled', 'unlabeled', 'both', 'none']
        self.label_rescale = label_rescale
        assert lam_bias in ['labeled', 'unlabeled', 'rand']
        self.lam_bias = lam_bias
        self.freeze_bn_unlabeled = freeze_bn_unlabeled

        self.loss_weights = loss_weights
        for key in loss_weights.keys():
            if not isinstance(loss_weights[key], list):
                self.loss_weights[key] = float(loss_weights[key]) \
                    if float(loss_weights[key]) > 0 else 0
        self.weight_pgc = loss_weights.get("weight_pgc", 1.)
        self.weight_one = loss_weights.get("weight_one", 1.)
        self.weight_mix_ll = loss_weights.get("weight_mix_ll", 1.)
        self.weight_mix_lu = loss_weights.get("weight_mix_lu", 1.)
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine
        # init
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q.init_weights(pretrained=pretrained)
        self.projector_q.init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)

        # create the momentum queue
        self.register_buffer("queue_list", torch.randn(
            proj_dim, queue_size * self.class_num))
        self.queue_list = nn.functional.normalize(self.queue_list, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(
            self.class_num, dtype=torch.long)) # pointer
    
    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        # init q
        if pretrained is not None:
            print_log('load encoder_q from: {}'.format(pretrained), logger='root')
        self.encoder_q.init_weights(pretrained=pretrained)
        self.projector_q.init_weights(init_linear='kaiming')
        if self.head_one is not None:
            self.head_one.init_weights(init_linear='normal')
        if self.head_mix is not None:
            self.head_mix.init_weights(init_linear='normal')
        # init k
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _freeze_bn(self):
        """ keep normalization layer freezed. """
        for m in self.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                m.eval()

    def _unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                m.train()

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.encoder_q(img)
        return x

    def mixup(self, img, gt_labels, dist_mode=False):
        """ mixup based on img and gt_labels
        
        Args:
            img (tensor): Input x with the size of (N, 2, C, H, W) or (N, C, H, W).
            gt_label (tensor): Ground truth labels of x in (N,).
            dist_mode (bool): Whether to mix with distributed mode.
        
        Returns:
            x (tensor): Encoded features from the backbone or raw mixed imgs.
            gt_label (tensor): Mixed labels for x.
        """
        # lam biased
        lam = np.random.beta(self.alpha, self.alpha)
        if self.lam_bias == 'labeled':
            lam = max(lam, 1-lam)
        elif self.lam_bias == 'unlabeled':
            lam = min(lam, 1-lam)
        # various mixup methods
        if self.mix_mode not in ["manifoldmix"]:
            if self.mix_mode in ["mixup", "cutmix", "saliencymix"]:
                img, gt_labels = eval(self.mix_mode)(img, gt_labels, lam=lam, dist_mode=dist_mode)
            elif self.mix_mode in ["resizemix", "fmix"]:
                mix_args = dict(lam=lam, dist_mode=dist_mode, **self.mix_args[self.mix_mode])
                img, gt_labels = eval(self.mix_mode)(img, gt_labels, **mix_args)
            else:
                raise NotImplementedError
            x = self.encoder_q(img)[-1]
        else:
            rand_index, _layer, _mask, gt_labels = self._manifoldmix(img, gt_labels)
            # manifoldmix
            if img.dim() == 5:
                cross_view = True
                img = img.reshape(-1, img.size(2), img.size(3), img.size(4))
            else:
                cross_view = False

            # args for mixup encoder_q
            mix_args = dict(
                layer=_layer, cross_view=cross_view, mask=_mask,
                BN_shuffle=False, idx_shuffle_BN=None,
                idx_shuffle_mix=rand_index, dist_shuffle=False)
            x = self.encoder_q(img, mix_args)[-1]
        return x, gt_labels

    def _manifoldmix(self, img, gt_label, lam=None):
        """ pixel-wise manifoldmix for the latent space mixup backbone """
        # manifoldmix
        if lam is None:
            lam = np.random.beta(self.alpha, self.alpha)
        
        rand_index = torch.randperm(img.size(0)).cuda()
        # mixup labels
        y_a = gt_label
        y_b = gt_label[rand_index]
        gt_label = (y_a, y_b, lam)
        
        _layer = np.random.randint(
            min(self.mix_args[self.mix_mode]["layer"]),
            max(self.mix_args[self.mix_mode]["layer"]), dtype=int)
        # generate mixup mask, should be [N, 1, H, W]
        _mask = None
        if img.size(3) > 64:  # normal version of resnet
            scale_factor = 2**(1 + _layer) if _layer > 0 else 1
        else:  # CIFAR version
            scale_factor = 2**(_layer - 1) if _layer > 1 else 1
        _mask_size = img.size(3) // scale_factor
        _mask = torch.zeros(img.size(0), 1, _mask_size, _mask_size).cuda()
        _mask[:] = lam

        return rand_index, _layer, _mask, gt_label

    def _update_loss_weights(self):
        """ update loss weights according to the cos_annealing scalar """
        # cos annealing decent, from 1 to 0
        for attr in self.loss_weights["decent_weight"]:
            setattr(self, attr, self.loss_weights[attr] * self.cos_annealing)
        # cos annealing accent, from 0 to 1
        for attr in self.loss_weights["accent_weight"]:
            setattr(self, attr, self.loss_weights[attr] * (1-self.cos_annealing))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key (EMA) encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key_c, c):
        """ Update queue of the class """
        # gather keys before updating queue
        batch_size = key_c.shape[0]
        ptr = int(self.queue_ptr[c])
        real_ptr = ptr + c * self.queue_size
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_list[:, real_ptr:real_ptr + batch_size] = key_c.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[c] = ptr

    def forward_pgc(self, im_q, im_k, labels):
        """ forward of PGC loss in Self-Tuning """
        batch_size = im_q.size(0)
        # compute query features
        q_f = self.encoder_q(im_q)[-1] 
        q_c = self.projector_q([q_f])[0] # queries: q_c (N x projector_dim)
        q_c = nn.functional.normalize(q_c, dim=1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_f = self.encoder_k(im_k)[-1] 
            k_c = self.projector_k([k_f])[0] # keys: k_c (N x projector_dim)
            k_c = nn.functional.normalize(k_c, dim=1)

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nl,nl->n', [q_c, k_c]).unsqueeze(-1)

        # cur_queue_list: queue_size * class_num
        cur_queue_list = self.queue_list.clone().detach()

        l_neg_list = torch.Tensor([]).cuda()
        l_pos_list = torch.Tensor([]).cuda()

        for i in range(batch_size):
            neg_sample = torch.cat([cur_queue_list[:, 0:labels[i]*self.queue_size],
                                    cur_queue_list[:, (labels[i]+1)*self.queue_size:]],
                                   dim=1)
            pos_sample = cur_queue_list[:, labels[i]*self.queue_size: (labels[i]+1)*self.queue_size]
            ith_neg = torch.einsum('nl,lk->nk', [q_c[i: i+1], neg_sample])
            ith_pos = torch.einsum('nl,lk->nk', [q_c[i: i+1], pos_sample])
            l_neg_list = torch.cat((l_neg_list, ith_neg), dim=0)
            l_pos_list = torch.cat((l_pos_list, ith_pos), dim=0)
            self._dequeue_and_enqueue(k_c[i: i+1], labels[i])
        
        # logits: 1 + queue_size + queue_size * (class_num - 1)
        PGC_logits = torch.cat([l_pos, l_pos_list, l_neg_list], dim=1)
        # apply temperature
        PGC_logits = nn.LogSoftmax(dim=1)(PGC_logits / self.temperature)

        PGC_labels = torch.zeros([batch_size, 1 + self.queue_size*self.class_num]).cuda()
        PGC_labels[:, 0:self.queue_size+1].fill_(1.0 / (self.queue_size + 1))
        return PGC_logits, PGC_labels, q_f
    
    def forward_train(self, img, gt_labels, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, 4, C, H, W). The first two are
                labeled while the latter two are unlabeled, which are mean centered
                and std scaled.
            gt_labels (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5 and img.size(1) == 4, \
            "Input both must have 5 dims, got: {} and {}".format(img.dim(), img.size(1))
        self._update_loss_weights()
        # ============= labeled data =============
        img_labeled_q = img[:, 0, ...].contiguous()
        img_labeled_k = img[:, 1, ...].contiguous()
        # 1.1 labeled PGC loss
        PGC_logit_labeled, PGC_label_labeled, _ = \
            self.forward_pgc(img_labeled_q, img_labeled_k, gt_labels)
        PGC_loss_labeled = self.KL(PGC_logit_labeled, PGC_label_labeled)

        # 1.2 onehot cls
        one_loss = None
        if self.head_one is not None and self.weight_one > 0:
            x = self.encoder_q(img_labeled_q)[-1]
            outs = self.head_one([x])
            one_loss = self.head_one.loss(outs, gt_labels)

        img_labeled_q = img[:, 0, ...].contiguous()
        # 1.3 mixup between labeled data
        mix_loss_ll = None
        if self.head_mix is not None and self.weight_mix_ll > 0:
            mixed_x, mixed_labels = self.mixup(img_labeled_q, gt_labels)
            outs = self.head_mix([mixed_x])
            mix_loss_ll = self.head_mix.loss(outs, mixed_labels)
        
        # ============= unlabeled data =============
        img_unlabeled_q = img[:, 2, ...].contiguous()
        img_unlabeled_k = img[:, 3, ...].contiguous()
        with torch.no_grad():  # no gradient for q
            q_f_unlabeled = self.encoder_q(img_unlabeled_q)[-1]
            if self.head_one is not None:
                logit_unlabeled = self.head_one([q_f_unlabeled])[0]
            elif self.head_mix is not None:
                logit_unlabeled = self.head_mix([q_f_unlabeled])[0]
            else:
                raise NotImplementedError
            # pseudo labels
            prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
            _, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)
        # 2.1 unlabeled PGC loss
        PGC_logit_unlabeled, PGC_label_unlabeled, _ = \
            self.forward_pgc(img_unlabeled_q, img_unlabeled_k, predict_unlabeled)
        PGC_loss_unlabeled = self.KL(PGC_logit_unlabeled, PGC_label_unlabeled)
        # 2.2 mixup between labeled and unlabeled
        mix_loss_lu = None
        if self.head_mix is not None and self.weight_mix_lu > 0:
            if self.freeze_bn_unlabeled:
                self._freeze_bn()
            img_lu = torch.cat(
                (img_labeled_q.unsqueeze(1), img_unlabeled_q.unsqueeze(1)), axis=1)
            mixed_x, mixed_labels = self.mixup(img_lu, gt_labels)
            outs = self.head_mix([mixed_x])
            if self.freeze_bn_unlabeled:
                self._unfreeze_bn()
            mixed_labels = (gt_labels, predict_unlabeled, mixed_labels[2])
            if self.label_rescale != 'both':
                if self.label_rescale == 'unlabeled':
                    label_mask = (False, True)
                elif self.label_rescale == 'labeled':
                    label_mask = (True, False)
                else:
                    label_mask = (False, False)
            else:
                label_mask = (True, True)
            loss_inputs = (outs, mixed_labels, label_mask)
            mix_loss_lu = self.head_mix.loss(*loss_inputs)
        
        # losses
        losses = dict()
        losses['loss'] = self.weight_pgc * (PGC_loss_labeled + PGC_loss_unlabeled)
        if one_loss is not None:
            losses['loss'] += self.weight_one * one_loss['loss']
            losses['acc_one'] = one_loss['acc']
        if mix_loss_ll is not None:
            losses['loss'] += self.weight_mix_ll * mix_loss_ll['loss']
            losses['acc_mix_ll'] = mix_loss_ll['acc_mix']
        if mix_loss_lu is not None:
            losses['loss'] += self.weight_mix_lu * mix_loss_lu['loss']
            losses['acc_mix_lu'] = mix_loss_lu['acc_mix']
        return losses

    def forward_test(self, img, **kwargs):
        """ original classification test """
        x = self.forward_backbone(img)  # tuple
        keys = []
        preds = []
        if self.head_one is not None:
            keys.append('acc_one')
            preds.append(self.head_one(x))
        if self.head_mix is not None:
            keys.append('acc_mix')
            preds.append(self.head_mix(x))
        out_tensors = [out[0].cpu() for out in preds]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_inference(self, img, **kwargs):
        """ inference prediction """
        x = self.encoder_q(img)
        preds = self.head_one(x, post_process=True)
        return preds[0]
