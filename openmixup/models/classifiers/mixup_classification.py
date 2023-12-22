import random
import logging
import numpy as np
import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32, load_checkpoint
from openmixup.utils import print_log
from torch.autograd import Variable

from .base_model import BaseModel
from .. import builder
from ..registry import MODELS
from ..augments import (cutmix, fmix, gridmix, mixup, resizemix, saliencymix, smoothmix,
                        alignmix, attentivemix, puzzlemix, snapmix, transmix)
from ..utils import PlotTensor


@MODELS.register_module
class MixUpClassification(BaseModel):
    """MixUp classification.

    Args:
        backbone (dict): Config dict for module of a backbone architecture.
        head (dict): Config dict for module of loss functions.
        backbone_k (dict, optional): Config dict for pre-trained backbone. Default: None.
        mix_block (dict, optional): Config dict for mix_block in AutoMix/SAMix.
        alpha (float or list): To sample Beta distribution in MixUp methods. Build a
            list for various mixup methods. Default: 1.
        mix_mode (str or list): Basice mixUp methods in input space. Similarly, build
            a list for various mix_mode, and randomly choose one mix_mode for each iter.
            Default: "mixup".
        mix_args (dict): Dict of args (hyper-parameters) for various mixup methods.
        mix_prob (list, optional): List of applying prob for given mixup modes. Default: None.
        mix_repeat (bool or int, optional): How many time to repeat mixup within a mini-batch.
            If mix_repeat > 1, mixup with different alpha and shuffle idx. Default: False.
        momentum_k (float, optional): Momentum update from the backbone k. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        pretrained_k (str, optional): Path to pre-trained weights for backbone_k or
            mix_block. Default: None.
        save_by_sample (bool): Whether to save mixup samples separately.
        debug_mode (bool): Whether to save some intermediate products.
    """

    def __init__(self,
                 backbone,
                 head=None,
                 backbone_k=None,
                 mix_block=None,
                 alpha=1.0,
                 mix_mode="mixup",
                 mix_args=dict(),
                 mix_prob=None,
                 mix_repeat=False,
                 momentum_k=-1,
                 pretrained=None,
                 pretrained_k=None,
                 cosine_update=False,
                 save_name='MixedSamples',
                 save_by_sample=False,
                 debug_mode=True,
                 init_cfg=None,
                 **kwargs):
        super(MixUpClassification, self).__init__(init_cfg, **kwargs)
        # networks
        assert isinstance(backbone, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)
        self.mix_block = None
        self.backbone_k = None
        self.momentum_k = momentum_k
        if backbone_k is not None:
            self.backbone_k = builder.build_backbone(backbone_k)
            for param in self.backbone_k.parameters():  # stop grad k
                param.requires_grad = False
            self.momentum_k = min(momentum_k, 1)
        if mix_block is not None:
            self.mix_block = builder.build_head(mix_block)
            for param in self.mix_block.parameters():  # stop grad mixblock
                param.requires_grad = False
        self.cosine_update = cosine_update
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine

        # mixup args
        self.mix_mode = mix_mode if isinstance(mix_mode, list) else [str(mix_mode)]
        self.dynamic_mode = {
            "alignmix": alignmix, "attentivemix": attentivemix, "puzzlemix": puzzlemix,
            "automix": self._mixblock, "samix": self._mixblock,
            "transmix": transmix,  # label mixup methods
            "snapmix": snapmix,
        }
        self.static_mode = {
            "mixup": mixup, "cutmix": cutmix, "fmix": fmix, "gridmix": gridmix,
            "manifoldmix": self._manifoldmix, "saliencymix": saliencymix,
            "smoothmix": smoothmix, "resizemix": resizemix,
        }
        self.mix_args = dict(  # default settings
            alignmix=dict(eps=0.1, max_iter=100),
            attentivemix=dict(grid_size=32, top_k=6, beta=8),
            automix=dict(mask_adjust=0, lam_margin=0),
            cutmix=dict(),
            fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False),
            gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
                cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
            mixup=dict(),
            manifoldmix=dict(layer=(0, 3)),
            puzzlemix=dict(transport=True, t_batch_size=None, block_num=5, beta=1.2,
                gamma=0.5, eta=0.2, neigh_size=4, n_labels=3, t_eps=0.8, t_size=4),
            snapmix=dict(),
            resizemix=dict(scope=(0.1, 0.8), use_alpha=False),
            # recursivemix=dict(old_img=None, old_label=None, num_classes=100, smoothing=0.0),
            saliencymix=dict(),
            samix=dict(mask_adjust=0, lam_margin=0.08),
            smoothmix=dict(),
            transmix=dict(mix_mode="cutmix"),
            vanilla=dict(),
        )
        _supported_mode = ["vanilla"] + list(self.dynamic_mode.keys()) + list(self.static_mode.keys())
        for _mode in _supported_mode:
            self.mix_args[_mode].update(mix_args.get(_mode, dict()))  # update mix_args
            if _mode == "manifoldmix":
                _layer_ = self.mix_args[_mode]["layer"]
                assert 0 <= min(_layer_) and max(_layer_) < 4
            if _mode == "resizemix":
                _scope_ = self.mix_args[_mode]["scope"]
                assert 0 <= min(_scope_) and max(_scope_) <= 1
        for _mode in self.mix_mode:
            assert _mode in _supported_mode, "The mix_mode={} is not supported!".format(_mode)
        self.alpha = alpha if isinstance(alpha, list) else [float(alpha)]
        assert len(self.alpha) == len(self.mix_mode) and len(self.mix_mode) < 6
        self.idx_list = [i for i in range(len(self.mix_mode))]
        self.mix_prob = mix_prob if isinstance(mix_prob, list) else None
        if self.mix_prob is not None:
            assert len(self.mix_prob) == len(self.alpha) and abs(sum(self.mix_prob)-1e-10) <= 1, \
                "mix_prob={}, sum={}, alpha={}".format(self.mix_prob, sum(self.mix_prob), self.alpha)
        self.mix_repeat = int(mix_repeat) if int(mix_repeat) > 1 else 1
        if self.mix_repeat > 1:
            print_log("Warning: mix_repeat={} is more than once.".format(self.mix_repeat))
        if len(self.mix_mode) < self.mix_repeat:
            print_log("Warning: the number of mix_mode={} is less than mix_repeat={}.".format(
                self.mix_mode, self.mix_repeat))
        self.debug_mode = debug_mode
        self.save_by_sample = save_by_sample
        self.save_name = str(save_name)
        self.save = False
        self.ploter = PlotTensor(apply_inv=True)
        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights. Default: None.
            pretrained_k (str, optional): Path to pre-trained weights for encoder_k.
                Default: None.
        """
        if self.init_cfg is not None:
            super(MixUpClassification, self).init_weights()

        # init pre-trained params
        if pretrained_k is not None:
            print_log('load pre-training from: {}'.format(pretrained_k), logger='root')
            if self.mix_block is not None and self.backbone_k is not None:
                load_checkpoint(self, pretrained_k, strict=False, logger=logging.getLogger())
            if self.mix_block is None and self.backbone_k is not None:
                self.backbone_k.init_weights(pretrained=pretrained_k)
        # init trainable params
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
            self.backbone.init_weights(pretrained=pretrained)
            self.head.init_weights()
        if self.backbone_k is not None and pretrained_k is None:
            for param_q, param_k in zip(self.backbone.parameters(),
                                        self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the backbone_k form backbone """
        # we don't update q to k when momentum when m<0
        if self.momentum_k < 0:
            return
        for param_q, param_k in zip(self.backbone.parameters(),
                                    self.backbone_k.parameters()):
            if self.momentum_k >= 1:
                param_k.data.copy_(param_q.data)
            else:
                param_k.data = param_k.data * self.momentum_k + \
                            param_q.data * (1. - self.momentum_k)

    @torch.no_grad()
    def _attribute_update(self):
        """Update some attributes in the backbone and head """
        if self.cosine_update:
            update_attr = getattr(self.backbone, 'update_attribute', None)
            if update_attr is not None:
                update_attr(self.cos_annealing)
            update_attr = getattr(self.head, 'update_attribute', None)
            if update_attr is not None:
                update_attr(self.cos_annealing)

    def _features(self, img, gt_label=None, cur_mode="puzzlemix", **kwargs):
        """ generating feature maps or gradient maps """
        if cur_mode == "attentivemix":
            img = F.interpolate(img,
                scale_factor=kwargs.get("feat_size", 224) / img.size(2), mode="bilinear")
            features = self.backbone_k(img)[0]
        elif cur_mode == "puzzlemix":
            input_var = Variable(img, requires_grad=True)
            self.backbone.eval()
            self.head.eval()
            pred = self.head(self.backbone(input_var))
            loss = self.head.loss(pred, gt_label)["loss"]
            loss.backward(retain_graph=False)
            features = torch.sqrt(torch.mean(input_var.grad**2, dim=1))  # grads
            # clear grads in models
            self.backbone.zero_grad()
            self.head.zero_grad()
            # return to train
            self.backbone.train()
            self.head.train()
        elif cur_mode == "snapmix":
            b, c, h, w = img.size()
            self.backbone.eval()
            self.head.eval()
            features = self.backbone(img)[-1]
            weight = self.head.fc.weight.data
            bias = self.head.fc.bias.data

            self.backbone.zero_grad()
            self.head.zero_grad()
            self.backbone.train()
            self.head.train()

            return (features, weight, bias)
        
        return features

    @torch.no_grad()
    def _mixblock(self):
        """ forward pre-trained mixblock """
        raise NotImplementedError

    @torch.no_grad()
    def _manifoldmix(self, img, gt_label, alpha, cur_mode="manifoldmix"):
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
            self.mix_args[cur_mode]["layer"][0], self.mix_args[cur_mode]["layer"][1], dtype=int)
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
            candidate_list = self.idx_list.copy()
            if 0 <= remove_idx <= len(self.idx_list):
                candidate_list.remove(int(remove_idx))
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
            cur_idx = random_state.choice(candidate_list, p=self.mix_prob)
        cur_mode, cur_alpha = self.mix_mode[cur_idx], self.alpha[cur_idx]

        # selecting label mixup methods
        label_mix_mode = "default"
        return_mask, mask = False, None  # return sample mixup mask in [N, 1, H, W]
        if cur_mode == "transmix":
            label_mix_mode, return_mask = "transmix", True
            cur_mode = self.mix_args["transmix"].get("mix_mode", "cutmix")  # sample mixup mode

        # applying dynamic sample mixup methods
        if cur_mode in ["attentivemix", "automix", "puzzlemix", "samix", "snapmix"]:
            if cur_mode in ["attentivemix", "puzzlemix", "snapmix"]:
                features = self._features(
                    img, gt_label=gt_label, cur_mode=cur_mode, **self.mix_args[cur_mode])
                mix_args = dict(alpha=cur_alpha, dist_mode=False, return_mask=return_mask,
                                features=features, **self.mix_args[cur_mode])
                img, gt_label = self.dynamic_mode[cur_mode](img, gt_label, **mix_args)
            elif cur_mode in ["automix", "samix"]:
                img = self._mixblock()
                raise NotImplementedError
            if return_mask:
                img, mask = img  # (img, mask): get mixup mask
            x = self.backbone(img)
        elif cur_mode == "alignmix":
            assert return_mask == False
            x = self.backbone(img)[-1]  # using the last layer
            mix_args = dict(alpha=cur_alpha, dist_mode=False, **self.mix_args[cur_mode])
            feat, gt_label = alignmix(x, gt_label, **mix_args)
            x = [feat]

        # applying hand-crafted sample mixup methods
        elif cur_mode not in ["manifoldmix",]:
            if cur_mode in ["mixup", "cutmix", "saliencymix", "smoothmix",]:
                img, gt_label = self.static_mode[cur_mode](
                    img, gt_label, cur_alpha, dist_mode=False, return_mask=return_mask)
            elif cur_mode in ["resizemix", "fmix", "gridmix",]:
                mix_args = dict(alpha=cur_alpha, dist_mode=False, return_mask=return_mask,
                                **self.mix_args[cur_mode])
                img, gt_label = self.static_mode[cur_mode](img, gt_label, **mix_args)
            else:
                assert cur_mode == "vanilla" and return_mask == False
            if return_mask:
                img, mask = img  # (img, mask): get mixup mask
            x = self.backbone(img)
        else:
            # manifoldmix
            assert return_mask == False
            rand_index, _layer, _mask, gt_label = self._manifoldmix(img, gt_label, cur_alpha)
            # args for mixup backbone
            mix_args = dict(
                layer=_layer, cross_view=False, mask=_mask, BN_shuffle=False, idx_shuffle_BN=None,
                idx_shuffle_mix=rand_index, dist_shuffle=False)
            x = self.backbone(img, mix_args)

        # applying label mixup methods
        if label_mix_mode == "transmix":
            assert mask is not None, "TransMix requires pre-defined sample mixup mask"
            y_a, y_b, lam0 = gt_label  # (y_a, y_b, lam): get lam
            x, cls_token, attn = x[-1]  # using the last layer feature and attention map
            patch_shape = (x.size(2), x.size(3))  # feature shape
            mix_args = dict(dist_mode=False, lam=lam0, attn=attn, patch_shape=patch_shape, mask=mask)
            _, gt_label = transmix(img, gt_label=y_a, **mix_args)
            gt_label = (y_a, y_b, gt_label[2])  # (y_a, y_b, lam'): update lam
            x = [[x, cls_token]]
        else:
            pass

        # save mixed img
        if self.save:
            plot_lam = gt_label[2] if len(gt_label) == 3 else None
            self.plot_mix(img_mixed=img, mix_mode=cur_mode, lam=plot_lam)
        # mixup loss
        outs = self.head(x)
        losses = self.head.loss(outs, gt_label)
        losses['loss'] /= self.mix_repeat
        if self.debug_mode:
            if torch.any(torch.isnan(losses['loss'])) or torch.any(torch.isinf(losses['loss'])):
                raise ValueError("Inf or nan value: use FP32 instead.")

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
        # before train
        with torch.no_grad():
            self._momentum_update()
            self._attribute_update()
        if isinstance(img, list):
            img = img[0]
        
        # repeat mixup aug within a mini-batch
        losses = dict()
        remove_idx = -1
        for i in range(self.mix_repeat):
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

        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.backbone(img)[-1:]  # classifying with the last layer
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def augment_test(self, img):
        """Test function with test time augmentation."""
        x = [self.backbone(_img)[-1] for _img in img]
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_test(self, img, **kwargs):
        """
        Args:
            img (List[Tensor] or Tensor): the outer list indicates the
                test-time augmentations and inner Tensor should have a
                shape of (N, C, H, W).
        """
        if isinstance(img, list):
            return self.augment_test(img)
        else:
            return self.simple_test(img)

    @force_fp32(apply_to=('img_mixed', ))
    def plot_mix(self, img_mixed, mix_mode="", lam=None):
        """ visualize mixup results """
        img = torch.cat((img_mixed[:4], img_mixed[4:8], img_mixed[8:12]), dim=0)
        title_name = "{}, lam={}".format(mix_mode, str(round(lam, 6))) \
            if isinstance(lam, float) else mix_mode
        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=4, title_name=title_name,
            save_name=self.save_name, make_grid=not self.save_by_sample)
