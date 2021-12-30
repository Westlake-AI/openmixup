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
        v09.14

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        head (dict): Config dict for module of loss functions. Default: None.
        alpha (float): To sample Beta distribution in MixUp methods.
        mix_mode (str): Basice mixUp methods in input space. Default: "mixup".
        mix_args (dict): Args for manifoldmix, resizeMix, fmix mode.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 head=None,
                 alpha=1.0,
                 mix_mode="mixup",
                 mix_args=dict(
                    manifoldmix=dict(layer=(0, 3)),
                    resizemix=dict(scope=(0.1, 0.8), use_alpha=False),
                    fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False)
                 ),
                 pretrained=None):
        super(MixUpClassification, self).__init__()
        assert mix_mode in ["mixup", "manifoldmix", "cutmix", "saliencymix", "resizemix", "fmix"]
        if mix_mode in ["manifoldmix"]:
            assert 0 == min(mix_args[mix_mode]["layer"]) and max(mix_args[mix_mode]["layer"]) < 4
        if mix_mode == "resizemix":
            assert 0 <= min(mix_args[mix_mode]["scope"]) and max(mix_args[mix_mode]["scope"]) <= 1
        self.backbone = builder.build_backbone(backbone)
        self.head = head
        self.mix_mode = mix_mode
        self.alpha = alpha
        self.mix_args = mix_args
        if head is not None:
            self.head = builder.build_head(head)
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

    def _manifoldmix(self, img, gt_label):
        """ pixel-wise manifoldmix for the latent space mixup backbone """
        # manifoldmix
        lam = np.random.beta(self.alpha, self.alpha)
        bs = img.size(0)
        rand_index = torch.randperm(bs).cuda()
        # mixup labels
        y_a = gt_label
        y_b = gt_label[rand_index]
        gt_label = (y_a, y_b, lam)
        
        _layer = np.random.randint(
            min(self.mix_args[self.mix_mode]["layer"]), max(self.mix_args[self.mix_mode]["layer"]), dtype=int)
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
        if self.mix_mode not in ["manifoldmix"]:
            if self.mix_mode in ["mixup", "cutmix", "saliencymix"]:
                img, gt_label = eval(self.mix_mode)(img, gt_label, self.alpha, dist_mode=False)
            elif self.mix_mode in ["resizemix", "fmix"]:
                mix_args = dict(alpha=self.alpha, dist_mode=False, **self.mix_args[self.mix_mode])
                img, gt_label = eval(self.mix_mode)(img, gt_label, **mix_args)
            else:
                raise NotImplementedError
            x = self.backbone(img)
        else:
            # manifoldmix
            rand_index, _layer, _mask, gt_label = self._manifoldmix(img, gt_label)
            
            # args for mixup backbone
            mix_args = dict(
                layer=_layer, cross_view=False, mask=_mask,
                BN_shuffle=False, idx_shuffle_BN=None, idx_shuffle_mix=rand_index, dist_shuffle=False)
            x = self.backbone(img, mix_args)
        
        outs = self.head(x)
        
        loss_inputs = (outs, gt_label)
        losses = self.head.loss(*loss_inputs)
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
