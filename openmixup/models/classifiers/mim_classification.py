import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from mmcv.runner import force_fp32
from openmixup.utils import print_log
from openmixup.models.utils import Canny, Laplacian, Sobel

from .base_model import BaseModel
from .. import builder
from ..registry import MODELS
from ..augments import (cutmix, fmix, gridmix, mixup, resizemix, saliencymix, smoothmix,
                     attentivemix, puzzlemix)
from ..utils import PlotTensor


@MODELS.register_module
class MIMClassification(BaseModel):
    """Image Classification with Mixups and MIM.

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck_cls (dict): Config dict for neck of classification pooling.
        neck_mim (dict): Config dict for neck of masked image modeling (MIM) decoder.
        head_cls (dict): Config dict for head of classification loss functions.
        head_mim (dict): Config dict for head of MIM loss functions.
        backbone_k (dict): Config dict for pre-trained backbone. Default: None.
        mim_target (None or str): Mode of MIM target. Default: None.
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
        pretrained_k (str, optional): Path to pre-trained weights for backbone_k.
            Default: None.
        loss_weights (dict): Loss weights of classification and MIM losses.
    """

    def __init__(self,
                 backbone,
                 neck_cls=None,
                 neck_mim=None,
                 head_cls=None,
                 head_mim=None,
                 backbone_k=None,
                 mim_target=None,
                 residual=False,
                 alpha=1.0,
                 mix_mode="mixup",
                 mix_args=dict(
                    attentivemix=dict(grid_size=32, top_k=6, beta=8),
                    automix=dict(mask_adjust=0, lam_margin=0),
                    fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False),
                    manifoldmix=dict(layer=(0, 3)),
                    puzzlemix=dict(transport=True, t_batch_size=None, block_num=5, beta=1.2,
                        gamma=0.5, eta=0.2, neigh_size=4, n_labels=3, t_eps=0.8, t_size=4),
                    resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
                    samix=dict(mask_adjust=0, lam_margin=0.08),
                 ),
                 mix_prob=None,
                 mix_repeat=False,
                 momentum_k=-1,
                 pretrained=None,
                 pretrained_k=None,
                 save_name="MIMcls",
                 loss_weights=dict(
                    decent_weight=[], accent_weight=[],
                    weight_mim=1, weight_cls=1,),
                 init_cfg=None,
                 **kwargs):
        super(MIMClassification, self).__init__(init_cfg, **kwargs)

        # networks
        self.backbone = builder.build_backbone(backbone)
        assert isinstance(neck_cls, dict) and isinstance(neck_mim, dict)
        self.neck_cls = builder.build_neck(neck_cls)
        self.neck_mim = builder.build_neck(neck_mim)
        assert isinstance(head_cls, dict) and isinstance(head_mim, dict)
        self.head_cls = builder.build_head(head_cls)
        self.head_mim = builder.build_head(head_mim)
        self.head = self.head_cls
        self.backbone_k = None
        self.momentum_k = momentum_k
        if backbone_k is not None:
            self.backbone_k = builder.build_backbone(backbone_k)
            for param in self.backbone_k.parameters():  # stop grad k
                param.requires_grad = False
            self.momentum_k = min(momentum_k, 1)
        
        # mim targets
        self.mim_target = mim_target
        self.residual = residual
        assert self.mim_target in [None, 'canny', 'hog', 'laplacian', 'lbp', 'pretrained', 'sobel',]
        if self.mim_target == 'canny':
            self.feat_layer = Canny(non_max_suppression=True, edge_smooth=True)
        elif self.mim_target == 'laplacian':
            self.feat_layer = Laplacian(mode='DoG', use_threshold=False)
        elif self.mim_target == 'sobel':
            self.feat_layer = Sobel(isotropic=True, use_threshold=False, out_channels=2)
        
        # mixup args
        self.mix_mode = mix_mode if isinstance(mix_mode, list) else [str(mix_mode)]
        for _mode in self.mix_mode:
            assert _mode in [
                "vanilla", "mixup", "manifoldmix",
                "cutmix", "fmix", "saliencymix", "smoothmix", "resizemix",
                "attentivemix", "puzzlemix", ]
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
            print_log("Warning: mix_repeat={} is more than once.".format(self.mix_repeat))
        if len(self.mix_mode) < self.mix_repeat:
            print_log("Warning: the number of mix_mode={} is less than mix_repeat={}.".format(
                self.mix_mode, self.mix_repeat))
        
        # save plots
        self.save_name = save_name
        self.save = False
        self.ploter = PlotTensor(apply_inv=True)
        # loss weights
        self.loss_weights = loss_weights
        for key in loss_weights.keys():
            if not isinstance(loss_weights[key], list):
                self.loss_weights[key] = float(loss_weights[key]) \
                    if float(loss_weights[key]) > 0 else 0
        self.weight_cls = loss_weights.get("weight_cls", 1.)
        self.weight_mim = loss_weights.get("weight_mim", 1.)
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine
        
        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights. Default: None.
            pretrained_k (str, optional): Path to pre-trained weights for encoder_k.
                Default: None.
        """
        # init pre-trained params
        if pretrained_k is not None:
            print_log('load pre-training from: {}'.format(pretrained_k), logger='root')
            if self.backbone_k is not None:
                self.backbone_k.init_weights(pretrained=pretrained_k)
        # init trainable params
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck_cls.init_weights()
        self.neck_mim.init_weights()
        self.head_cls.init_weights()
        self.head_mim.init_weights()
        if self.backbone_k is not None and pretrained_k is None:
            for param_q, param_k in zip(self.backbone.parameters(),
                                        self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _update_loss_weights(self):
        """ update loss weights according to the cos_annealing scalar """
        # cos annealing decent, from 1 to 0
        for attr in self.loss_weights["decent_weight"]:
            setattr(self, attr, self.loss_weights[attr] * self.cos_annealing)
        # cos annealing accent, from 0 to 1
        for attr in self.loss_weights["accent_weight"]:
            setattr(self, attr, self.loss_weights[attr] * (1-self.cos_annealing))

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

    def _features(self, img, gt_label=None, cur_mode="puzzlemix", **kwargs):
        """ generating feature maps or gradient maps """
        if cur_mode == "attentivemix":
            img = F.interpolate(img,
                scale_factor=kwargs.get("feat_size", 224) / img.size(2), mode="bilinear")
            features = self.backbone_k(img)[-1]
        elif cur_mode == "puzzlemix":
            input_var = Variable(img, requires_grad=True)
            self.backbone.eval()
            self.head_cls.eval()
            pred = self.neck_cls([self.backbone(input_var)[-1]])
            pred = self.head_cls(pred)
            loss = self.head_cls.loss(pred, gt_label)["loss"]
            loss.backward(retain_graph=False)
            features = torch.sqrt(torch.mean(input_var.grad**2, dim=1))  # grads
            # clear grads in models
            self.backbone.zero_grad()
            self.head_cls.zero_grad()
            # return to train
            self.backbone.train()
            self.head_cls.train()
        
        return features

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

    def forward_mix(self, img, gt_label, mask=None, remove_idx=-1):
        """computate mini-batch mixup.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.
            mask (tensor): MIM mask.
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
        outputs = dict(cur_idx=cur_idx)
        cur_mode, cur_alpha = self.mix_mode[cur_idx], self.alpha[cur_idx]
        
        # applying dynamic methods
        if cur_mode in ["attentivemix", "automix", "puzzlemix", "samix",]:
            if cur_mode in ["attentivemix", "puzzlemix"]:
                features = self._features(
                    img, gt_label=gt_label, cur_mode=cur_mode, **self.mix_args[cur_mode])
                mix_args = dict(alpha=cur_alpha, dist_mode=False,
                                features=features, **self.mix_args[cur_mode])
                img, gt_label = eval(cur_mode)(img, gt_label, **mix_args)
            feat = self.backbone(img, mask)
        # hand-crafted methods
        elif cur_mode not in ["manifoldmix",]:
            if cur_mode in ["mixup", "cutmix", "saliencymix", "smoothmix",]:
                img, gt_label = eval(cur_mode)(img, gt_label, cur_alpha, dist_mode=False)
            elif cur_mode in ["resizemix", "fmix"]:
                mix_args = dict(alpha=cur_alpha, dist_mode=False, **self.mix_args[cur_mode])
                img, gt_label = eval(cur_mode)(img, gt_label, **mix_args)
            else:
                assert cur_mode == "vanilla"
            feat = self.backbone(img, mask)
        else:
            # manifoldmix
            rand_index, _layer, _mask, gt_label = self._manifoldmix(img, gt_label, cur_alpha)
            
            # args for mixup backbone
            mix_args = dict(
                layer=_layer, cross_view=False, mask=_mask, BN_shuffle=False, idx_shuffle_BN=None,
                idx_shuffle_mix=rand_index, dist_shuffle=False)
            # TODO: Not support ManifoldMix now
            feat = self.backbone(img, mix_args)
        outputs['feat'] = feat
        
        # save mixed img
        if self.save and cur_mode != "vanilla":
            plot_lam = gt_label[2] if len(gt_label) == 3 else None
            self.plot_mix(img_mixed=img, mix_mode=cur_mode, lam=plot_lam)
        # mixup loss
        outs = self.neck_cls([feat[-1]], mask)
        outs = self.head_cls(outs)
        losses = self.head_cls.loss(outs, gt_label)
        losses['loss'] *= (self.weight_cls / self.mix_repeat)

        return losses, outputs

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
            mask (torch.Tensor): MIM mask of shape (N, H, W).
            gt_label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # before train
        with torch.no_grad():
            self._update_loss_weights()
            self._momentum_update()
        
        # raw img and MIM targets
        mask = kwargs.get('mask', None)
        if isinstance(mask, list):
            mask, img_mim = mask
        else:
            img_mim = img.clone()
        if self.mim_target in ['canny', 'laplacian', 'sobel',]:
            assert img_mim.size(1) == 3
            img_mim = self.feat_layer(img_mim)
        elif self.mim_target == 'pretrained':
            img_mim = self.backbone_k(img_mim)[-1]
        
        # repeat mixup aug within a mini-batch
        losses = dict()
        remove_idx = -1
        for i in range(int(self.mix_repeat)):
            # Notice: cutmix related methods need 'inplace operation' on Variable img,
            #   thus we use 'img.clone()' for each iteration.
            if i == 0:
                losses, outputs = self.forward_mix(
                    img.clone(), gt_label, mask=mask, remove_idx=remove_idx)
            else:
                _loss, outputs = self.forward_mix(
                    img.clone(), gt_label, mask=mask, remove_idx=remove_idx)
                losses["loss"] += _loss["loss"]
            remove_idx = outputs['cur_idx']
            # remove 'vanilla' if chosen
            if self.mix_mode[outputs['cur_idx']] == "vanilla":
                # apply mim
                img_rec = self.neck_mim(outputs['feat'])
                if isinstance(img_rec, list):
                    img_rec = img_rec[-1]
                if self.residual:
                    img_rec += img_mim.mean(dim=(2, 3), keepdim=True).expand(img_rec.size())
                losses["mim"] = self.head_mim(x=img_mim, x_rec=img_rec, mask=mask)["loss"]
                losses["loss"] += (losses["mim"] * self.weight_mim)
                # save mim
                if self.save:
                    self.plot_reconstruction(img, img_mim, img_rec, mask)
        
        return losses

    def forward_test(self, img, **kwargs):
        x = self.neck_cls([self.backbone(img)[-1]])  # tuple
        outs = self.head_cls(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    @force_fp32(apply_to=('img', 'img_mim', 'img_rec', 'mask',))
    def plot_reconstruction(self, img, img_mim, img_rec, mask):
        """ visualize reconstruction results """
        nrow = 4
        img_mim = img_mim[:nrow]
        img_rec = img_rec[:nrow]
        img = img[:nrow]
        img_raw = None
        plot_args = dict(dpi=None, apply_inv=True, overwrite=False)
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
        mim_save_name = self.save_name.split(".png")[0] + "_mim.png"
        mim_title = self.mim_target if self.mim_target is not None else 'rgb'
        self.ploter.plot(
            img, nrow=nrow, title_name="MIM"+mim_title, save_name=mim_save_name, **plot_args)

    @force_fp32(apply_to=('img_mixed',))
    def plot_mix(self, img_mixed, mix_mode="", lam=None):
        """ visualize mixup results """
        img = torch.cat((img_mixed[:4], img_mixed[4:8], img_mixed[8:12]), dim=0)
        title_name = "{}, lam={}".format(mix_mode, str(round(lam, 6))) \
            if isinstance(lam, float) else mix_mode
        assert self.save_name.find(".png") != -1
        save_name_mix = self.save_name.split(".png")[0] + "_mix.png"
        self.ploter.plot(
            img, nrow=4, title_name=title_name, save_name=save_name_mix)
