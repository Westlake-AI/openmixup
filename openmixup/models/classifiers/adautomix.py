import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional
import PIL.Image
import random
import logging
from mmcv.runner import auto_fp16, force_fp32, load_checkpoint
from torchvision.utils import save_image
from openmixup.utils import print_log
from .base_model import BaseModel
from .. import builder
from ..registry import MODELS
from ..augments import cutmix, mixup
from ..utils import PlotTensor


@MODELS.register_module
class AdAutoMix(BaseModel):
    def __init__(self,
                 backbone,
                 backbone_k=None,
                 mix_block=None,
                 affine_da=None,
                 head_mix=None,
                 head_one=None,
                 head_mix_k=None,
                 head_one_k=None,
                 head_weights=dict(decent_weight=[], accent_weight=[],
                                   head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
                 alpha=1.0,
                 mix_samples=3,
                 is_random=False,
                 momentum=0.999,
                 lam_margin=-1,
                 switch_off=0.,
                 adv_interval=2,
                 mixup_radio=0.5,
                 beta_radio=0.3,
                 head_one_mix=False,
                 head_ensemble=False,
                 save=False,
                 save_name='MixedSamples',
                 debug=False,
                 mix_shuffle_no_repeat=False,
                 pretrained=None,
                 pretrained_k=None,
                 init_cfg=None,
                 **kwargs):
        super(AdAutoMix, self).__init__(init_cfg, **kwargs)
        # basic params
        self.alpha = float(alpha)
        self.mix_samples = int(mix_samples)
        self.co_mix = 2
        self.is_random = bool(is_random)
        self.momentum = float(momentum)
        self.base_momentum = float(momentum)
        self.lam_margin = float(lam_margin) if float(lam_margin) > 0 else 0
        self.mixup_radio = float(mixup_radio)
        self.beta_radio = float(beta_radio)
        self.switch_off = float(switch_off) if float(switch_off) > 0 else 0
        self.head_one_mix = bool(head_one_mix)
        self.head_ensemble = bool(head_ensemble)
        self.save = bool(save)
        self.save_name = str(save_name)
        self.ploter = PlotTensor(apply_inv=True)
        self.debug = bool(debug)
        self.mix_shuffle_no_repeat = bool(mix_shuffle_no_repeat)
        self.adv_interval = int(adv_interval)
        self.iter = 0
        self.i = 0
        assert 0 <= self.momentum and self.lam_margin < 1

        # network
        assert isinstance(mix_block, dict) and isinstance(backbone, dict)
        # assert isinstance(affine_da, dict) and isinstance(backbone, dict)
        assert backbone_k is None or isinstance(backbone_k, dict)
        assert head_mix is None or isinstance(head_mix, dict)
        assert head_one is None or isinstance(head_one, dict)
        assert head_mix_k is None or isinstance(head_mix_k, dict)
        assert head_one_k is None or isinstance(head_one_k, dict)
        head_mix_k = head_mix if head_mix_k is None else head_mix_k
        head_one_k = head_one if head_one_k is None else head_one_k
        # mixblock
        self.mix_block = builder.build_head(mix_block)
        # backbone
        self.backbone_q = builder.build_backbone(backbone)
        if backbone_k is not None:
            self.backbone_k = builder.build_backbone(backbone_k)
            assert self.momentum >= 1. and pretrained_k is not None
        else:
            self.backbone_k = builder.build_backbone(backbone)
        self.backbone = self.backbone_k  # for feature extract
        # mixup cls head
        assert "head_mix_q" in head_weights.keys() and "head_mix_k" in head_weights.keys()
        self.head_mix_q = builder.build_head(head_mix)
        self.head_mix_k = builder.build_head(head_mix_k)
        # onehot cls head
        if "head_one_q" in head_weights.keys():
            self.head_one_q = builder.build_head(head_one)
        else:
            self.head_one_q = None
        if "head_one_k" in head_weights.keys():
            self.head_one_k = builder.build_head(head_one_k)
        else:
            self.head_one_k = None
        # for feature extract
        self.head = self.head_one_k if self.head_one_k is not None else self.head_one_q
        # onehot and mixup heads for training
        self.weight_mix_q = head_weights.get("head_mix_q", 1.)
        self.weight_mix_k = head_weights.get("head_mix_k", 1.)
        self.weight_one_q = head_weights.get("head_one_q", 1.)
        assert self.weight_mix_q > 0 and (self.weight_mix_k > 0 or backbone_k is not None)
        self.head_weights = head_weights
        self.head_weights['decent_weight'] = head_weights.get("decent_weight", list())
        self.head_weights['accent_weight'] = head_weights.get("accent_weight", list())
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine

        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):

        # init mixblock
        if self.mix_block is not None:
            self.mix_block.init_weights(init_linear='normal')
        # init pretrained backbone_k and mixblock
        if pretrained_k is not None:
            print_log('load pretrained classifier k from: {}'.format(pretrained_k), logger='root')
            # load full ckpt to backbone and fc
            logger = logging.getLogger()
            load_checkpoint(self, pretrained_k, strict=False, logger=logger)

        # init backbone, based on params in q
        if pretrained is not None:
            print_log('load encoder_q from: {}'.format(pretrained), logger='root')
        self.backbone_q.init_weights(pretrained=pretrained)
        # copy backbone param from q to k
        if pretrained_k is None and self.momentum < 1:
            for param_q, param_k in zip(self.backbone_q.parameters(),
                                        self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False  # stop grad k

        # init head
        if self.head_mix_q is not None:
            self.head_mix_q.init_weights()
        if self.head_one_q is not None:
            self.head_one_q.init_weights()

        # copy head one param from q to k
        if (self.head_one_q is not None and self.head_one_k is not None) and \
                (pretrained_k is None and self.momentum < 1):
            for param_one_q, param_one_k in zip(self.head_one_q.parameters(),
                                                self.head_one_k.parameters()):
                param_one_k.data.copy_(param_one_q.data)
                param_one_k.requires_grad = False  # stop grad k

        # copy head mix param from q to k
        if (self.head_mix_q is not None and self.head_mix_k is not None) and \
                (pretrained_k is None and self.momentum < 1):
            for param_mix_q, param_mix_k in zip(self.head_mix_q.parameters(),
                                                self.head_mix_k.parameters()):
                param_mix_k.data.copy_(param_mix_q.data)
                param_mix_k.requires_grad = False  # stop grad k

    def _update_loss_weights(self):
        """ update loss weights according to the cos_annealing scalar """
        if self.cos_annealing < 0 or self.cos_annealing > 1:
            return
        # cos annealing decent, from 1 to 0
        if len(self.head_weights["decent_weight"]) > 0:
            for attr in self.head_weights["decent_weight"]:
                setattr(self, attr, self.head_weights.get(attr, 1.) * self.cos_annealing)
        # cos annealing accent, from 0 to 1
        if len(self.head_weights["accent_weight"]) > 0:
            for attr in self.head_weights["accent_weight"]:
                setattr(self, attr, self.head_weights.get(attr, 1.) * (1 - self.cos_annealing))

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the k form q by hook, including the backbone and heads """
        # we don't update q to k when momentum > 1
        if self.momentum >= 1.:
            return
        # update k's backbone and cls head from q
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

        if self.head_one_q is not None and self.head_one_k is not None:
            for param_one_q, param_one_k in zip(self.head_one_q.parameters(),
                                                self.head_one_k.parameters()):
                param_one_k.data = param_one_k.data * self.momentum + \
                                    param_one_q.data * (1 - self.momentum)

        if self.head_mix_q is not None and self.head_mix_k is not None:
            for param_mix_q, param_mix_k in zip(self.head_mix_q.parameters(),
                                                self.head_mix_k.parameters()):
                param_mix_k.data = param_mix_k.data * self.momentum + \
                                   param_mix_q.data * (1 - self.momentum)

    def _no_repeat_shuffle_idx(self, batch_size_this, ignore_failure=False):
        """ generate no repeat shuffle idx within a gpu """
        idx_shuffle = torch.randperm(batch_size_this).cuda()
        idx_original = torch.tensor([i for i in range(batch_size_this)]).cuda()
        idx_repeat = False
        for i in range(10):  # try 10 times
            if (idx_original == idx_shuffle).any() == True:
                idx_repeat = True
                idx_shuffle = torch.randperm(batch_size_this).cuda()
            else:
                idx_repeat = False
                break
        # hit: prob < 1.2e-3
        if idx_repeat == True and ignore_failure == False:
            # way 2: repeat prob = 0, but too simple!
            idx_shift = np.random.randint(1, batch_size_this-1)
            idx_shuffle = torch.tensor(  # shift the original idx
                [(i+idx_shift) % batch_size_this for i in range(batch_size_this)]).cuda()
        return idx_shuffle

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of a batch of images, (N, C, H, W).
            gt_label (Tensor): Groundtruth onehot labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if isinstance(img, list):
            img = img[0]
        batch_size = img.size(0)
        self._update_loss_weights()

        if self.mix_shuffle_no_repeat:
            index_bb = self._no_repeat_shuffle_idx(batch_size, ignore_failure=True)
            index_mb = self._no_repeat_shuffle_idx(batch_size, ignore_failure=False)
        else:
            index_bb = torch.randperm(batch_size).cuda()
            index_mb = torch.randperm(batch_size).cuda()

        # choose mixed samples
        if self.is_random:
            self.co_mix = random.randint(2, self.mix_samples)
        else:
            self.co_mix = self.mix_samples
        lam = np.random.beta(self.alpha, self.alpha, self.co_mix)  # 0: mb, 1: bb

        # Mixup
        co_lam = []
        index = [index_mb, index_bb]
        feature = self.backbone_k(img)[0]
        # co-samples mix
        if self.co_mix > 2:
            # index
            for i in range(2, self.co_mix):
                co_mb = torch.randperm(batch_size).cuda()
                index.append(co_mb)
                index_mb = index
                index_bb = [torch.randperm(batch_size).cuda() for i in range(0, len(index_mb))]
            # lam
            for i in range(0, self.co_mix):
                co_lam.append(lam[i] / lam.sum())
            results = self.co_mixup(img, co_lam, index_mb, index_bb, feature)

            # save img mb
            self.co_plot_mix(results["img_mix_mb"], img[index_mb[0], :], img[index_mb[1], :], img[index_mb[2], :],
                             co_lam, results["debug_plot"], "co-mix")
            # k (mb): the mix block training
            cos_simi_weight, loss_mix_k = self.forward_k(img, results["img_mix_mb"], gt_label, index_mb, co_lam)
            # q (bb): the encoder training
            loss_one_q, loss_mix_q = self.forward_q(img, results["img_mix_bb"], gt_label, index_bb, co_lam)

        # two samples mix
        else:
            results = self.mixup(img, lam, index, feature)
            # save img mb
            if self.save:
                self.plot_mix(results["img_mix_mb"], img, img[index_mb, :], lam, results["debug_plot"], "mixblock")
            # k (mb): the mix block training
            cos_simi_weight, loss_mix_k = self.forward_k(img, results["img_mix_mb"], gt_label, index[0], lam[0])
            # q (bb): the encoder training
            loss_one_q, loss_mix_q = self.forward_q(img, results["img_mix_bb"], gt_label, index[1], lam[1])

        #  loss summary
        losses = {
            'loss': loss_mix_q['loss'] * self.weight_mix_q,
            'acc_mix_q': loss_mix_q['acc'],
        }
        # onehot loss
        if loss_one_q is not None and self.weight_one_q > 0:
            losses['loss'] += loss_one_q['loss'] * self.weight_one_q
            losses['acc_one_q'] = loss_one_q['acc']

        # adversial training
        if self.iter % self.adv_interval == 0:
            if loss_mix_k['loss'] is not None:
                loss_mix_k['loss'] = -1.0 * self.beta_radio * loss_mix_k['loss'] \
                                    + (1 - self.beta_radio) * cos_simi_weight
            self.iter = 0
        # mixblock loss
        if loss_mix_k['loss'] is not None and self.weight_mix_k > 0:
            losses["loss"] += loss_mix_k['loss'] * self.weight_mix_k
            losses['acc_mix_k'] = loss_mix_k['acc']
        else:
            losses['acc_mix_k'] = loss_mix_q['acc']
        self.iter += 1
        return losses

    @force_fp32(apply_to=('im_mixed', 'im_q', 'im_k',))
    def plot_mix(self, im_mixed, im_q, im_k, lam, debug_plot=None, name="k"):
        """ visualize mixup results, supporting 'debug' mode """
        # plot mixup results
        img = torch.cat((im_q[:4], im_k[:4], im_mixed[:4]), dim=0)
        title_name = 'lambda {}={}'.format(name, lam)
        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=4, title_name=title_name, save_name=self.save_name)

        # debug: plot intermediate results, fp32
        if self.debug:
            assert isinstance(debug_plot, dict)
            for key, value in debug_plot.items():
                _, h, w = value.size()
                img = value[:4].view(h, 4 * w).type(torch.float32).detach().cpu().numpy()
                fig = plt.figure()
                plt.imshow(img)
                # plt.title('debug {}, lambda k={}'.format(str(key), lam))
                _debug_path = self.save_name.split(".png")[0] + "_{}.png".format(str(key))
                if not os.path.exists(_debug_path):
                    plt.savefig(_debug_path, bbox_inches='tight')
                plt.close()

    @force_fp32(apply_to=('im_mixed', 'im_1', 'im_2', 'im_3'))
    def co_plot_mix(self, im_mixed, im_1, im_2, im_3, lam, debug_plot=None, name="k" ):
        """ visualize mixup results, supporting 'debug' mode """
        # plot mixup results
        img = torch.cat((im_1[:4], im_2[:4], im_3[:4], im_mixed[:4]), dim=0)
        title_name = 'lambda {}={}'.format(name, lam)
        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=4, title_name=title_name, save_name=self.save_name)

        # debug: plot intermediate results, fp32
        if self.debug:
            assert isinstance(debug_plot, dict)
            for key, value in debug_plot.items():
                _, h, w = value.size()
                img = value[:4].view(h, 4 * w).type(torch.float32).detach().cpu().numpy()
                fig = plt.figure()
                plt.imshow(img)
                # plt.title('debug {}, lambda k={}'.format(str(key), lam))
                _debug_path = self.save_name.split(".png")[0] + "_{}.png".format(str(key))
                if not os.path.exists(_debug_path):
                    plt.savefig(_debug_path, bbox_inches='tight')
                plt.close()

    @auto_fp16(apply_to=('x', 'mixed_x',))
    def forward_q(self, x, mixed_x, y, index, lam):
        """
        Args:
            x (Tensor): Input of a batch of images, (N, C, H, W).
            mixed_x (Tensor): Mixup images of x, (N, C, H, W).
            y (Tensor): Groundtruth onehot labels, coresponding to x.
            index (List): Input list of shuffle index (tensor) for mixup.
            lam (List): Input list of lambda (scalar).

        Returns:
            dict[str, Tensor]: loss_one_q and loss_mix_q are losses from q.
        """
        loss_one_q = None
        if self.head_one_q is not None and self.weight_one_q > 0:
            if self.head_one_mix:  # mixup in head_one
                if np.random.random() > 0.5:
                    x, y_one_mix = mixup(x.clone(), y, 0.8, dist_mode=False)
                else:
                    x, y_one_mix = cutmix(x.clone(), y, 1.0, dist_mode=False)
            out_one_q = self.backbone_q(x)[-1]
            pred_one_q = self.head_one_q([out_one_q])
            # loss
            if self.head_one_mix:
                loss_one_q = self.head_one_q.loss(pred_one_q, y_one_mix)
            else:
                loss_one_q = self.head_one_q.loss(pred_one_q, y)
            if torch.isnan(loss_one_q['loss']):
                print_log("Warming NAN in loss_one_q. Please use FP32!", logger='root')
                loss_one_q = None

        # co_mixup q
        loss_co_mix_q = None
        if self.co_mix > 2:
            co_mixed_x = x[index[0], ] * lam[0]
            for i in range(1, len(lam)):
                co_mixed_x += x[index[i], ] * lam[i]
            co_out_mix_q = self.backbone_q(co_mixed_x)[-1]
            pred_co_mix_q = self.head_mix_q([co_out_mix_q])
            # loss
            y_co_mix_q = []
            for i in range(0, self.co_mix):
                y_co_mix_q.append(y[index[i]])
            y_co_mix_q.append(lam)
            y_co_mix_q = tuple(y_co_mix_q)
            loss_co_mix_q = self.head_mix_q.co_loss(pred_co_mix_q, y_co_mix_q)
            if torch.isnan(loss_co_mix_q['loss']):
                print_log("Warming NAN in loss_co_mix_q. Please use FP32!", logger='root')
                loss_co_mix_q = dict(loss=None)
            # loss re-weight
            loss_re_q = dict(loss=None, acc=None)
            loss_re_q['loss'] = loss_one_q['loss'] * (1 - self.mixup_radio) + loss_co_mix_q['loss'] * self.mixup_radio
            loss_re_q['acc'] = (loss_one_q['acc'] + loss_co_mix_q['acc']) / 2
        else:
            loss_re_q = loss_one_q

        # mixup q
        loss_mix_q = None
        if self.weight_mix_q > 0:
            out_mix_q = self.backbone_q(mixed_x)[-1]
            pred_mix_q = self.head_mix_q([out_mix_q])
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_mix_q[0] = pred_mix_q[0].type(torch.float32)
            # mixup loss
            if self.co_mix == 2:
                y_mix_q = (y, y[index], lam)
                loss_mix_q = self.head_mix_q.loss(pred_mix_q, y_mix_q)
            else:
                y_mix_q = []
                for i in range(0, self.co_mix):
                    y_mix_q.append(y[index[i]])
                y_mix_q.append(lam)
                y_mix_q = tuple(y_mix_q)
                loss_mix_q = self.head_mix_q.co_loss(pred_mix_q, y_mix_q)
            if torch.isnan(loss_mix_q['loss']):
                print_log("Warming NAN in loss_mix_q. Please use FP32!", logger='root')
                loss_mix_q = dict(loss=None)

        return loss_re_q, loss_mix_q

    @auto_fp16(apply_to=('mixed_x',))
    def forward_k(self, x, mixed_x, y, index, lam):
        """ forward k with the mixup sample """
        loss_mix_k = dict(loss=None, pre_mix_loss=None)
        cos_simi_weight = 0.0
        # switch off mixblock training
        if self.switch_off > 0:
            if 0 < self.cos_annealing <= 1:
                if np.random.rand() > self.switch_off * self.cos_annealing:
                    return cos_simi_weight, loss_mix_k

        # training mixblock from k
        if self.weight_mix_k > 0:
            # mixed_x forward
            out_mix_k = self.backbone_k(mixed_x)
            pred_mix_k = self.head_mix_k([out_mix_k[-1]])
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_mix_k[0] = pred_mix_k[0].type(torch.float32)
            # mixup loss
            if self.co_mix == 2:
                y_mix_k = (y, y[index], lam)
                loss_mix_k = self.head_mix_q.loss(pred_mix_k, y_mix_k)
            else:
                y_mix_k = []
                for i in range(0, self.co_mix):
                    y_mix_k.append(y[index[i]])
                y_mix_k.append(lam)
                y_mix_k = tuple(y_mix_k)
                loss_mix_k = self.head_mix_q.co_loss(pred_mix_k, y_mix_k)
            if torch.isnan(loss_mix_k['loss']):
                print_log("Warming NAN in loss_mix_k. Please use FP32!", logger='root')
                loss_mix_k["loss"] = None

        # cosine similarty
        simi = torch.nn.CosineSimilarity(dim=-1)
        if self.co_mix == 2:
            x_ = x[index, :]
            pred_x = self.head_mix_k([self.backbone_k(x)[-1]])[0]
            pred_x_ = self.head_mix_k([self.backbone_k(x_)[-1]])[0]
            cos_simi_weight = torch.mean(simi(pred_mix_k[0], pred_x) * lam \
                              + simi(pred_mix_k[0], pred_x_) * (1-lam))
        elif self.co_mix > 2:
            pred = []
            for i in range(0, self.co_mix):
                pred.append(self.head_mix_k([self.backbone_k(x[index[i], :])[-1]])[0])

            for i in range(0, self.co_mix):
                cos_simi_weight += simi(pred_mix_k[0], pred[i]) * lam[i]

            cos_simi_weight = torch.mean(cos_simi_weight)
        return cos_simi_weight, loss_mix_k

    def co_mixup(self, x, lam, index_mb, index_bb, feature):
        results = dict()
        b, c, w, h = x.size()
        # lam info
        mb = []
        bb = []
        for i in range(0, len(index_mb)):
            mb.append(feature[index_mb[i], :])
            bb.append(feature[index_bb[i], :])

        # training mixblock
        mask_mb = self.mix_block(mb, lam)
        # training backbone
        mask_bb = self.mix_block(bb, lam)

        if self.debug:
            results["debug_plot"] = mask_mb["debug_plot"]
        else:
            results["debug_plot"] = None

        mask_mb = mask_mb["mask"]
        mask_bb = mask_bb["mask"].clone().detach()

        # lam_margin for backbone training, this for 3 samples mixing
        if self.lam_margin >= lam[0] or self.lam_margin < 1 - lam[0]:
            mask_bb[:, 0, :, :] = lam[0]
            mask_bb[:, 1, :, :] = lam[1]
            mask_bb[:, 2, :, :] = lam[2]

        # mix, apply mask on x and x_
        assert mask_mb.shape[1] == self.co_mix
        img_mix_mb = torch.zeros([b, c, w, h]).to(x)
        img_mix_bb = torch.zeros([b, c, w, h]).to(x)
        for i in range(0, self.co_mix):
            img_mix_mb += x[index_mb[i], :] * mask_mb[:, i, :, :].unsqueeze(1)
            img_mix_bb += x[index_bb[i], :] * mask_bb[:, i, :, :].unsqueeze(1)

        results["img_mix_mb"] = img_mix_mb
        results["img_mix_bb"] = img_mix_bb

        return results

    def mixup(self, x, lam, index, feature):
        """ pixel-wise input space mixup"""
        results = dict()
        # lam info
        lam_mb = lam[0]  # lam is a scalar
        lam_bb = lam[1]

        # get mixup mask
        mb = [feature, feature[index[0], :]]
        bb = [feature, feature[index[1], :]]

        mask_mb = self.mix_block(mb, lam_mb)
        mask_bb = self.mix_block(bb, lam_bb)

        if self.debug:
            results["debug_plot"] = mask_mb["debug_plot"]
        else:
            results["debug_plot"] = None

        mask_mb = mask_mb["mask"]
        mask_bb = mask_bb["mask"].clone().detach()

        # lam_margin for backbone training
        if self.lam_margin >= lam_bb or self.lam_margin >= 1 - lam_bb:
            mask_bb[:, 0, :, :] = lam_bb
            mask_bb[:, 1, :, :] = 1 - lam_bb

        # mix, apply mask on x and x_
        assert mask_mb.shape[1] == 2
        assert mask_mb.shape[2:] == x.shape[2:], f"Invalid mask shape={mask_mb.shape}"
        results["img_mix_mb"] = \
            x * mask_mb[:, 0, :, :].unsqueeze(1) + x[index[0], :] * mask_mb[:, 1, :, :].unsqueeze(1)
        results["img_mix_bb"] = \
            x * mask_bb[:, 0, :, :].unsqueeze(1) + x[index[1], :] * mask_bb[:, 1, :, :].unsqueeze(1)

        return results

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        keys = list()  # 'acc_mix_k', 'acc_one_k', 'acc_mix_q', 'acc_one_q'
        pred = list()
        # backbone
        last_k = self.backbone_k(img)[-1]
        last_q = self.backbone_q(img)[-1]
        # head k
        if self.weight_mix_k > 0:
            pred.append(self.head_mix_k([last_k]))
            keys.append('acc_mix_k')
            if self.head_one_k is not None:
                pred.append(self.head_one_k([last_k]))
                keys.append('acc_one_k')
        # head q
        pred.append(self.head_mix_q([last_q]))
        keys.append('acc_mix_q')
        if self.head_one_q is not None:
            pred.append(self.head_one_q([last_q]))
            keys.append('acc_one_q')
        # head ensemble
        if self.head_ensemble:
            pred.append([torch.stack(
                [pred[i][0] ** 2 for i in range(len(pred))]).mean(dim=0)])
            keys.append('acc_avg')

        out_tensors = [p[0].cpu() for p in pred]  # NxC
        return dict(zip(keys, out_tensors))

    def augment_test(self, img, **kwargs):
        """Test function with test time augmentation."""
        keys = list()  # 'acc_mix_k', 'acc_one_k', 'acc_mix_q', 'acc_one_q'
        pred = list()
        # backbone
        img = img[0]
        last_k = self.backbone_k(img)[-1]
        last_q = self.backbone_q(img)[-1]
        # head k
        if self.weight_mix_k > 0:
            pred.append(self.head_mix_k([last_k]))
            keys.append('acc_mix_k')
            if self.head_one_k is not None:
                pred.append(self.head_one_k([last_k]))
                keys.append('acc_one_k')
        # head q
        pred.append(self.head_mix_q([last_q]))
        keys.append('acc_mix_q')
        if self.head_one_q is not None:
            pred.append(self.head_one_q([last_q]))
            keys.append('acc_one_q')
        # head ensemble
        if self.head_ensemble:
            pred.append([torch.stack(
                [pred[i][0] ** 2 for i in range(len(pred))]).mean(dim=0)])
            keys.append('acc_avg')

        out_tensors = [p[0].cpu() for p in pred]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_test(self, img, **kwargs):
        """Forward computation during testing.

        Args:
            img (List[Tensor] or Tensor): the outer list indicates the
                test-time augmentations and inner Tensor should have a
                shape of (N, C, H, W).

        Returns:
            dict[key, Tensor]: A dictionary of head names (key) and predictions.
        """
        if isinstance(img, list):
            return self.augment_test(img)
        else:
            return self.simple_test(img)

    def forward_inference(self, img, **kwargs):
        """Forward output for inference.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.

        Returns:
            tuple[Tensor]: final model outputs.
        """
        x = self.backbone(img)[-1]
        preds = self.head([x], post_process=True)
        return preds[0]

    def forward_vis(self, img, gt_label, **kwargs):
        """" visualization by jupyter notebook """
        batch_size = img.size()[0]
        lam = kwargs.get('lam', [0.5, 0.5])
        index = [i + 1 for i in np.arange(batch_size)]
        index[-1] = 0

        # forward mixblock
        indices = [index, index]
        feature = self.backbone_k(img)[0]
        results = self.mixup(img, gt_label, lam, indices, feature)
        return {'mix_bb': results["img_mix_bb"], 'mix_mb': results["img_mix_mb"]}
