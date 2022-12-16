import random
import numpy as np
import torch
import torch.nn as nn

from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS
from ..augments import cutmix, fmix, gridmix, mixup, resizemix, saliencymix, smoothmix


@MODELS.register_module
class DMixMatch(BaseModel):
    """
    Implementation of DMixMatch (using Decoupled Mixup)
        based on FixMatch, FlexMatch and mixup methods.

    *** Requiring Hook: `momentum_update` is adjusted by `CosineScheduleHook`
        after_train_iter in `momentum_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        head (dict): Config dict for module of loss functions. Default: None.
        head_mix: Config dict for mixup classification head.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        momentum (float): Momentum coefficient for the EMA encoder. Default: 0.999.
        temperature (float): Temperature scaling parameter for output sharpening, only
            when hard_label = False). Default: 0.5.
        p_cutoff (float): Confidence cutoff hyper-parameter for unsupervised loss masking.
            Default: 0.95.
        weight_ul (float): Loss weight of unsupervised loss to supervised loss.
        hard_label (bool): If True, consistency regularization use a hard pseudo label.
        ratio_ul (float): Sample ratio of unlabeled v.s. labeled. Default: 7.
        ema_pseudo (float): Whether to generate pseudo labels by the EMA model. If the
            ema_pseudo > 0, it denotes the prob in (0,1]. Default: 1.0.
        deduplicate (bool): Whether to remove the duplicated samples in the labeled,
            espectially for the 400 labeled case in CIFAR. Default: False.
        alpha (float): To sample Beta distribution in MixUp methods.
        mix_mode (str): Basice mixUp methods in input space. Default: "mixup".
        mix_args (dict): Args for manifoldmix, resizeMix, fmix mode. Default: None.
        mix_prob (list): List of applying prob for given mixup modes. Default: None.
        p_mix_cutoff (float): Confidence cutoff hyper-parameter for the unsupervised
            mixup loss masking. Default: 0.95.
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
                 head=None,
                 head_mix=None,
                 momentum=0.999,
                 temperature=0.5,
                 p_cutoff=0.95,
                 weight_ul=1.0,
                 hard_label=True,
                 ratio_ul=7,
                 ema_pseudo=1.0,
                 alpha=1.0,
                 mix_mode="mixup",
                 mix_args=dict(
                    manifoldmix=dict(layer=(0, 3)),
                    resizemix=dict(scope=(0.1, 0.8), use_alpha=False),
                    fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False)
                 ),
                 mix_prob=None,
                 p_mix_cutoff=0.95,
                 label_rescale='labeled',
                 lam_bias='labeled',
                 loss_weights=dict(
                    decent_weight=[],
                    accent_weight=['weight_mix_lu'],
                    weight_one=1, weight_mix_ll=1, weight_mix_lu=1),
                 deduplicate=False,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(DMixMatch, self).__init__(init_cfg, **kwargs)
        # network settings
        self.encoder = nn.Sequential(
            builder.build_backbone(backbone), builder.build_head(head))
        self.encoder_k = nn.Sequential(  # EMA
            builder.build_backbone(backbone), builder.build_head(head))
        self.head_mix = None
        if head_mix is not None:
            self.head_mix = builder.build_head(head_mix)
            self.head_mix_k = builder.build_head(head_mix)
        self.init_weights(pretrained=pretrained)

        # FixMatch args
        self.momentum = float(momentum)
        self.base_momentum = float(momentum)
        self.temperature = float(temperature)
        self.p_cutoff = float(p_cutoff)
        self.weight_ul = float(weight_ul)
        self.hard_label = bool(hard_label)
        self.ratio_ul = int(ratio_ul)
        self.ema_pseudo = float(ema_pseudo)
        self.deduplicate = bool(deduplicate)
        assert 1 <= ratio_ul and 0 < p_cutoff <= 1

        # mixup args
        self.mix_mode = mix_mode if isinstance(mix_mode, list) else [str(mix_mode)]
        for _mode in self.mix_mode:
            assert _mode in ["vanilla", "mixup", "manifoldmix", "cutmix", "saliencymix", "resizemix", "fmix"]
            if _mode == "manifoldmix":
                assert 0 <= min(mix_args[_mode]["layer"]) and max(mix_args[_mode]["layer"]) < 4
            if _mode == "resizemix":
                assert 0 <= min(mix_args[_mode]["scope"]) and max(mix_args[_mode]["scope"]) <= 1
        self.alpha = alpha if isinstance(alpha, list) else [float(alpha)]
        assert len(self.alpha) == len(self.mix_mode) and len(self.mix_mode) < 6
        self.idx_list = [i for i in range(len(self.mix_mode))]
        self.mix_args = mix_args
        self.mix_prob = mix_prob if isinstance(mix_prob, list) else None
        if self.mix_prob is not None:
            assert len(self.mix_prob) == len(self.alpha) and abs(sum(self.mix_prob)-1e-10) <= 1, \
                "mix_prob={}, sum={}, alpha={}".format(self.mix_prob, sum(self.mix_prob), self.alpha)
            for i in range(1, len(self.mix_prob)):
                self.mix_prob[i] = self.mix_prob[i] + self.mix_prob[i-1]
        assert label_rescale in ['labeled', 'unlabeled', 'both', 'none']
        self.label_rescale = label_rescale
        assert lam_bias in ['labeled', 'unlabeled', 'rand']
        self.lam_bias = lam_bias
        assert 0 < p_mix_cutoff <= 1 and p_mix_cutoff <= p_cutoff
        self.p_mix_cutoff = float(p_mix_cutoff)

        self.loss_weights = loss_weights
        for key in loss_weights.keys():
            if not isinstance(loss_weights[key], list):
                self.loss_weights[key] = float(loss_weights[key]) \
                    if float(loss_weights[key]) > 0 else 0
        self.weight_one = loss_weights.get("weight_one", 1.)
        self.weight_mix_ll = loss_weights.get("weight_mix_ll", 1e-5)
        self.weight_mix_lu = loss_weights.get("weight_mix_lu", 1e-5)
        self.cos_annealing = 1 - 1e-5  # decent from 1 to 0 as cosine

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        # init encoder q
        if pretrained is not None:
            print_log('load encoder from: {}'.format(pretrained), logger='root')
        self.encoder[0].init_weights(pretrained=pretrained)
        self.encoder[1].init_weights(init_linear='normal')
        if self.head_mix is not None:
            self.head_mix.init_weights(init_linear='normal')
        # EMA
        for param_q, param_k in zip(self.encoder.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        if self.head_mix is not None:
            for param_q, param_k in zip(self.head_mix.parameters(),
                                        self.head_mix_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

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
        # choose a mixup method
        if self.mix_prob is None:
            candidate_list = self.idx_list.copy()
            cur_idx = random.choices(candidate_list, k=1)[0]
        else:
            rand_n = random.random()
            for i in range(len(self.idx_list)):
                if self.mix_prob[i] > rand_n:
                    cur_idx = self.idx_list[i]
                    break
        cur_mode, cur_alpha = self.mix_mode[cur_idx], self.alpha[cur_idx]
        # lam biased
        lam = np.random.beta(cur_alpha, cur_alpha)
        if self.lam_bias == 'labeled':
            lam = max(lam, 1-lam)
        elif self.lam_bias == 'unlabeled':
            lam = min(lam, 1-lam)

        # applying mixup methods
        if cur_mode not in ["manifoldmix"]:
            # Notice: cutmix related methods need 'inplace operation' on Variable img,
            #   thus we use 'img.clone()' for each iteration.
            if cur_mode in ["mixup", "cutmix", "saliencymix"]:
                img, gt_labels = eval(cur_mode)(img.clone(), gt_labels, lam=lam, dist_mode=dist_mode)
            elif cur_mode in ["resizemix", "fmix"]:
                mix_args = dict(lam=lam, dist_mode=dist_mode, **self.mix_args[cur_mode])
                img, gt_labels = eval(cur_mode)(img.clone(), gt_labels, **mix_args)
            else:
                assert cur_mode == "vanilla"
            x = self.encoder[0](img)[-1]
        else:
            rand_index, _layer, _mask, gt_labels = self._manifoldmix(img, gt_labels, lam=lam)
            # manifoldmix
            if img.dim() == 5:
                cross_view = True
                img = img.reshape(-1, img.size(2), img.size(3), img.size(4))
            else:
                cross_view = False
            # args for manifoldmix encoder_q
            mix_args = dict(
                layer=_layer, cross_view=cross_view, mask=_mask,
                BN_shuffle=False, idx_shuffle_BN=None,
                idx_shuffle_mix=rand_index, dist_shuffle=False)
            x = self.encoder[0](img, mix_args)[-1]
        return x, gt_labels

    def _manifoldmix(self, img, gt_label, lam=None, alpha=None):
        """ pixel-wise manifoldmix for the latent space mixup backbone """
        # manifoldmix
        if lam is None:
            lam = np.random.beta(alpha, alpha)
        
        rand_index = torch.randperm(img.size(0)).cuda()
        # mixup labels
        y_a = gt_label
        y_b = gt_label[rand_index]
        gt_label = (y_a, y_b, lam)
        
        _layer = np.random.randint(
            min(self.mix_args["manifoldmix"]["layer"]),
            max(self.mix_args["manifoldmix"]["layer"]), dtype=int)
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
        _cos_annealing = max(0, min(1, self.cos_annealing))  # [0, 1]
        for attr in self.loss_weights["decent_weight"]:
            setattr(self, attr, self.loss_weights[attr] * _cos_annealing)
        # cos annealing accent, from 0 to 1
        for attr in self.loss_weights["accent_weight"]:
            setattr(self, attr, self.loss_weights[attr] * (max(0, 1-_cos_annealing)))
        # mixup loss weight
        if self.head_mix is not None:
            self.weight_mix_ll = max(1e-5, self.weight_mix_ll)

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the EMA encoder."""
        for param_q, param_k in zip(self.encoder.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
        if self.head_mix is not None:
            for param_q, param_k in zip(self.head_mix.parameters(),
                                        self.head_mix_k.parameters()):
                if self.weight_mix_ll <= 1e-3 and self.weight_mix_lu < 1e-3:
                    param_k.data.copy_(param_q.data)
                else:
                    param_k.data = param_k.data * self.momentum + \
                                param_q.data * (1. - self.momentum)

    def forward_train(self, img, gt_labels, gt_idx, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, 3, C, H, W). The first sample is
                labeled while the latter two are unlabeled (weak, strong), which are
                mean centered and std scaled.
            gt_labels (Tensor): Ground-truth labels.
            gt_idx (Tensor): The labeled sample idx.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5 and img.size(1) == 3, \
            "Input both must have 5 dims, got: {} and {}".format(img.dim(), img.size(1))
        bs, _, c, h, w = img.size()
        # update loss weights
        self._update_loss_weights()
        
        # ============= labeled data =============
        if self.deduplicate:
            value, indices = gt_idx.sort()
            value_idx = torch.range(1, value.size(0)).cuda().type(torch.long) % value.size(0)
            mask_idx = (value - value[value_idx, ...]).type(torch.bool)  # remove repeat idx samples
            img_labeled = img[indices, 0, ...][mask_idx]
            gt_labels = gt_labels[indices][mask_idx]
            # shuffle the sorted samples
            rand_index = torch.randperm(img_labeled.size(0)).cuda()
            img_labeled = img_labeled[rand_index]
            gt_labels = gt_labels[rand_index]
            num_l = min(img.size(0) // self.ratio_ul, img_labeled.size(0))
            img_labeled = img_labeled[:num_l, ...]
        else:
            num_l = img.size(0) // self.ratio_ul
            img_labeled = img[:num_l, 0, ...].contiguous()
        # 1.1 head q one-hot cls
        pred_l = self.encoder(img_labeled)  # logits: N_lxC
        loss_l = self.encoder[1].loss(pred_l, gt_labels[:num_l, ...])
        # 1.2 mixup between labeled data
        loss_mix_ll = None
        if self.head_mix is not None and self.weight_mix_ll > 0:
            mixed_x, mixed_labels = self.mixup(
                img_labeled.reshape(num_l, c, h, w), gt_labels[:num_l, ...])
            # Notice: since mix_ll will cost low mask_ratio in the early stage, 
            #   thus we stop mix_lu to affect the backbone when < 1e-3
            if self.weight_mix_ll < 1e-3:
                mixed_x = mixed_x.detach()
            pred_mix_ll = self.head_mix([mixed_x])
            loss_mix_ll = self.head_mix.loss(pred_mix_ll, mixed_labels)

        # ============= unlabeled data =============
        if self.ema_pseudo > np.random.random():
            logits_ul_s = self.encoder(img[:, 2, ...].squeeze())[0]  # q logits: 2NxC
            logits_ul_w = self.encoder_k(img[:, 1, ...].squeeze())[0].detach()  # k logits: NxC
        else:
            img_unlabeled = img[:, 1:, ...].reshape(
                img.size(0) * 2, img.size(2), img.size(3), img.size(4))
            logits_ul = self.encoder(img_unlabeled)[0]  # logits: 2NxC
            logits_ul_w, logits_ul_s = logits_ul.chunk(2)
            logits_ul_w = logits_ul_w.detach()
        # pseudo labels
        pseudo_label = torch.softmax(logits_ul_w, dim=-1)  # NxC
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)  # Nx1
        mask_pl = max_probs.ge(self.p_cutoff).float()  # Nx1, mask out uncertain samples
        # 2.1 fixmatch unlabeled loss
        if self.hard_label == True:
            pseudo_label = max_idx
            loss_ul = self.encoder[1].loss([logits_ul_s], pseudo_label, weight=mask_pl)
        else:
            pseudo_label = torch.softmax(logits_ul_w / self.temperature, dim=-1)
            loss_ul = self.encoder[1].loss([logits_ul_s], pseudo_label, weight=mask_pl)
        # 2.2 mixup between labeled and unlabeled
        loss_mix_lu = None
        if self.head_mix is not None and self.weight_mix_lu > 1e-5:
            if self.deduplicate:
                mixed_x, mixed_labels = self.mixup(
                    torch.cat((img_labeled[:num_l, ...].unsqueeze(1),
                               img[:num_l, 2, ...].unsqueeze(1)), dim=1),
                    gt_labels[:num_l, ...])
            else:
                mixed_x, mixed_labels = self.mixup(
                    img[:num_l, :2, ...], gt_labels[:num_l, ...])
            # Notice: mix_lu is unreliable in the early stage and will cost low
            #   mask_ratio, thus we stop mix_lu to affect the backbone when < 1e-3
            if self.weight_mix_lu < 1e-3:
                mixed_x = mixed_x.detach()
            pred_mix_lu = self.head_mix([mixed_x])
            # Notice: we set lam for L and 1-lam for UL by default.
            mixed_labels = (  # y_a, y_b, lam
                gt_labels[:num_l, ...], pseudo_label[:num_l, ...], mixed_labels[2])
            if self.label_rescale != 'both':
                if self.label_rescale == 'unlabeled':
                    label_mask = (False, True)
                elif self.label_rescale == 'labeled':
                    label_mask = (True, False)
                else:
                    label_mask = (False, False)
            else:
                label_mask = (True, True)
            # mask for UL mixup loss, thr = min(p_cutoff, max(p_mix_cutoff, 1-lam))
            mask_mix = max_probs[:num_l, ...].ge(
                min(self.p_cutoff, max(self.p_mix_cutoff, 1-mixed_labels[2]))).float()  # Nx1
            loss_mix_lu = self.head_mix.loss(
                pred_mix_lu, mixed_labels, label_mask=label_mask, weight=mask_mix)
        
        # losses
        losses = {
            'loss': loss_l['loss'] * self.weight_one + \
                    loss_ul['loss'] * self.weight_ul,
            'mask_ratio': mask_pl.mean(),
            'acc_l': loss_l['acc'], 'acc_ul': loss_ul['acc'],
        }
        if loss_mix_ll is not None:
            losses['loss'] += loss_mix_ll['loss'] * self.weight_mix_ll
            losses['acc_mix_ll'] = loss_mix_ll['acc_mix']
        if loss_mix_lu is not None:
            losses['loss'] += loss_mix_lu['loss'] * self.weight_mix_lu
            losses['acc_mix_lu'] = loss_mix_lu['acc_mix']
        return losses

    def forward_test(self, img, **kwargs):
        """ Using EMA model to test """
        x = self.encoder_k[0](img)  # tuple
        keys = []
        preds = []
        keys.append('acc_one')
        preds.append(self.encoder_k[1](x))
        if self.head_mix is not None:
            keys.append('acc_mix')
            preds.append(self.head_mix(x))
        out_tensors = [out[0].cpu() for out in preds]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_inference(self, img, **kwargs):
        """ inference prediction """
        x = self.encoder_k[0](img)
        preds = self.encoder_k[1](x, post_process=True)
        return preds[0]
