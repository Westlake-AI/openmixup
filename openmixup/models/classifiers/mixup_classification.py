import random
import numpy as np
import torch
import torch.nn as nn

from openmixup.utils import print_log

from .. import builder
from ..registry import MODELS
from ..utils import cutmix, mixup, saliencymix, resizemix, fmix


@MODELS.register_module
class MixUpClassification(nn.Module):
    """MixUp classification.
        v01.09 (randomly selecting mix_mode)
        v01.17 (add mix_repeat)

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        head (dict): Config dict for module of loss functions.
        alpha (float or list): To sample Beta distribution in MixUp methods. Build a
            list for various mixup methods. Default: 1.
        mix_mode (str or list): Basice mixUp methods in input space. Similarly, build
            a list for various mix_mode, and randomly choose one mix_mode for each iter.
            Default: "mixup".
        mix_args (dict): Args for manifoldmix, resizeMix, fmix mode.
        mix_prob (list): List of applying prob for given mixup modes. Default: None.
        mix_repeat (bool or int): How many time to repeat mixup within a mini-batch. If
            mix_repeat > 1, mixup with different alpha and shuffle idx. Default: False.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 head,
                 alpha=1.0,
                 mix_mode="mixup",
                 mix_args=dict(
                    manifoldmix=dict(layer=(0, 3)),
                    resizemix=dict(scope=(0.1, 0.8), use_alpha=False),
                    fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False)
                 ),
                 mix_prob=None,
                 mix_repeat=False,
                 mix_reverse=False,
                 pretrained=None,
                 **kwargs):
        super(MixUpClassification, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)
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
        self.mix_repeat = int(mix_repeat) if int(mix_repeat) > 1 else 1
        if self.mix_repeat > 1:
            print_log("Warning: mix_repeat={} is more than 1!".format(self.mix_repeat))
        if len(self.mix_mode) < self.mix_repeat:
            print_log("Warning: the number of mix_mode={} is less than mix_repeat={}.".format(
                self.mix_mode, self.mix_repeat))
        self.mix_reverse = mix_reverse
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        if self.head is not None:
            self.head.init_weights()

    def _manifoldmix(self, img, gt_label, alpha, cur_mode='manifoldmix'):
        """ pixel-wise manifoldmix for the latent space mixup backbone """
        # manifoldmix
        lam = np.random.beta(alpha, alpha)
        bs = img.size(0)
        rand_index = torch.randperm(bs).cuda()
        # mixup labels
        y_a = gt_label
        y_b = gt_label[rand_index]
        gt_label = (y_a, y_b, lam)
        
        _layer = np.random.randint(
            min(self.mix_args[cur_mode]["layer"]), max(self.mix_args[cur_mode]["layer"]), dtype=int)
        # generate mixup mask
        _mask = None
        if img.size(3) > 64:  # normal version of resnet
            scale_factor = 2**(1 + _layer) if _layer > 0 else 1
        else:  # CIFAR version
            scale_factor = 2**(_layer - 1) if _layer > 1 else 1
        _mask_size = img.size(3) // scale_factor
        _mask = torch.zeros(img.size(0), 1, _mask_size, _mask_size).cuda()
        _mask[:] = lam

        return rand_index, _layer, _mask, gt_label

    def forward_mix(self, img, gt_label, remove_idx=-1):
        """computate mini-batch mixup.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.
            remove_idx (int): Remove this idx this time.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # choose a mixup method
        if self.mix_prob is None:
            candidate_list = self.idx_list.copy()
            if 0 <= remove_idx <= len(self.idx_list):
                candidate_list.remove(int(remove_idx))
            cur_idx = random.choices(candidate_list, k=1)[0]
        else:
            rand_n = random.random()
            for i in range(len(self.idx_list)):
                if self.mix_prob[i] > rand_n:
                    cur_idx = self.idx_list[i]
                    if cur_idx == remove_idx:  # randomly choose one among the rest
                        candidate_list = self.idx_list.copy()
                        candidate_list.remove(int(remove_idx))
                        cur_idx = random.choices(candidate_list, k=1)[0]
                    break
        cur_mode, cur_alpha = self.mix_mode[cur_idx], self.alpha[cur_idx]
        
        # applying mixup methods
        if cur_mode not in ["manifoldmix"]:
            if cur_mode in ["mixup", "cutmix", "saliencymix"]:
                img, gt_label = eval(cur_mode)(img, gt_label, cur_alpha, dist_mode=False)
                if self.mix_reverse:  # (y_a, y_b, lam)
                    gt_label = (gt_label[1], gt_label[0], gt_label[2])
            elif cur_mode in ["resizemix", "fmix"]:
                mix_args = dict(alpha=cur_alpha, dist_mode=False, **self.mix_args[cur_mode])
                img, gt_label = eval(cur_mode)(img, gt_label, **mix_args)
                if self.mix_reverse:  # (y_a, y_b, lam)
                    gt_label = (gt_label[1], gt_label[0], gt_label[2])
            else:
                assert cur_mode == "vanilla"
            x = self.backbone(img)
        else:
            # manifoldmix
            rand_index, _layer, _mask, gt_label = self._manifoldmix(img, gt_label, cur_alpha)
            
            # args for mixup backbone
            mix_args = dict(
                layer=_layer, cross_view=False, mask=_mask,
                BN_shuffle=False, idx_shuffle_BN=None, idx_shuffle_mix=rand_index, dist_shuffle=False)
            x = self.backbone(img, mix_args)
        
        outs = self.head(x)
        loss_inputs = (outs, gt_label)
        losses = self.head.loss(*loss_inputs)
        return losses, cur_idx

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # repeat mixup aug within a mini-batch
        losses = dict()
        remove_idx = -1
        for i in range(int(self.mix_repeat)):
            # Notice: cutmix related methods need 'inplace operation' on Variable img,
            #   thus we use 'img.clone()' for each iteration.
            if i == 0:
                losses, cur_idx = self.forward_mix(img.clone(), gt_label, remove_idx=remove_idx)
            else:
                _loss, cur_idx = self.forward_mix(img.clone(), gt_label, remove_idx=remove_idx)
                losses["loss"] += _loss["loss"]
            # remove 'vanilla' if chosen
            if self.mix_mode[cur_idx] == "vanilla":
                remove_idx = cur_idx
        losses["loss"] /= self.mix_repeat
        return losses

    def forward_test(self, img, **kwargs):
        x = self.backbone(img)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_calibration(self, img, **kwargs):
        img, gt_label = img[0], img[1]
        
        inputs = (img, False)
        x = self.backbone(inputs)
        outs = self.head(x)
        return outs

    def aug_test(self, imgs):
        raise NotImplementedError

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'calibration':
            return self.forward_calibration(img, **kwargs)
        elif mode == 'extract':
            return self.self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
