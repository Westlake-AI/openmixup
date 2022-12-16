import numpy as np
import torch
import torch.nn as nn

from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class FixMatch(BaseModel):
    """
    Implementation of "FixMatch: Simplifying Semi-Supervised Learning with Consistency
        and Confidence (https://arxiv.org/abs/2001.07685), NeurIPS, 2020."
        * Tensorflow (official): https://github.com/google-research/fixmatch
        * PyTorch (flexmatch): https://github.com/torchssl/torchssl

    *** Requiring Hook: `momentum_update` is adjusted by `CosineScheduleHook`
        after_train_iter in `momentum_hook.py`.
    
    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        momentum (float): Momentum coefficient for the EMA encoder. Default: 0.999.
        temperature (float): Temperature scaling parameter for output sharpening, only
            when hard_label = False). Default: 0.5.
        p_cutoff (float): Confidence cutoff parameters for loss masking. Default: 0.95.
        weight_ul (float): Loss weight of unsupervised loss to supervised loss.
        hard_label (bool): If True, consistency regularization use a hard pseudo label.
        ratio_ul (float): Sample ratio of unlabeled v.s. labeled. Default: 7.
        ema_pseudo (float): Whether to generate pseudo labels by the EMA model. If the
            ema_pseudo > 0, it denotes the prob in (0,1]. Default: 1.0.
        deduplicate (bool): Whether to remove the duplicated samples in the labeled,
            espectially for the 400 labeled case in CIFAR. Default: False.
    """

    def __init__(self,
                 backbone,
                 head=None,
                 momentum=0.999,
                 temperature=0.5,
                 p_cutoff=0.95,
                 weight_ul=1.0,
                 hard_label=True,
                 ratio_ul=7,
                 ema_pseudo=1.0,
                 deduplicate=False,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(FixMatch, self).__init__(init_cfg, **kwargs)
        self.encoder = nn.Sequential(
            builder.build_backbone(backbone), builder.build_head(head))
        self.encoder_k = nn.Sequential(  # EMA
            builder.build_backbone(backbone), builder.build_head(head))
        self.init_weights(pretrained=pretrained)

        self.momentum = float(momentum)
        self.base_momentum = float(momentum)
        self.temperature = float(temperature)
        self.p_cutoff = float(p_cutoff)
        self.weight_ul = float(weight_ul)
        self.hard_label = bool(hard_label)
        self.ratio_ul = int(ratio_ul)
        self.ema_pseudo = float(ema_pseudo)
        self.deduplicate = bool(deduplicate)
        assert ratio_ul >= 1 and p_cutoff <= 1

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
        # EMA
        for param_q, param_k in zip(self.encoder.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the EMA encoder."""
        for param_q, param_k in zip(self.encoder.parameters(),
                                    self.encoder_k.parameters()):
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
        
        # for labeled data
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

        # head q
        logits_l = self.encoder(img_labeled)  # logits: N_lxC
        loss_l = self.encoder[1].loss(logits_l, gt_labels[:num_l, ...])
        
        # for unlabeled data
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
        mask = max_probs.ge(self.p_cutoff).float()  # Nx1, mask out uncertain samples
        # select = max_probs.ge(self.p_cutoff).long()
        if self.hard_label == True:
            pseudo_label = max_idx
            loss_ul = self.encoder[1].loss([logits_ul_s], pseudo_label, weight=mask)
        else:
            pseudo_label = torch.softmax(logits_ul_w / self.temperature, dim=-1)
            loss_ul = self.encoder[1].loss([logits_ul_s], pseudo_label, weight=mask)
        
        # losses
        losses = {
            'loss': loss_l['loss'] + loss_ul['loss'] * self.weight_ul,
            'mask_ratio': mask.mean(),
            'acc_l': loss_l['acc'], 'acc_ul': loss_ul['acc'],
        }
        return losses

    def forward_test(self, img, **kwargs):
        """ Using EMA model to test """
        outs = self.encoder_k(img)  # k, NxC logits
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_inference(self, img, **kwargs):
        """ inference prediction """
        x = self.encoder_k[0](img)
        preds = self.encoder_k[1](x, post_process=True)
        return preds[0]
