import os
import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import logging
from contextlib import contextmanager, nullcontext
from mmcv.runner import auto_fp16, force_fp32, load_checkpoint
from ..augments import cutmix, mixup
from openmixup.utils import print_log
from .base_model import BaseModel
from .. import builder
from ..registry import MODELS
from ..utils import PlotTensor
# ------ Token Merge Functions ------ #
from ..utils.merge import parse_r, modify_r_list, check_parse_r, token_unmerge_from_map, token_unmerge
from ..backbones.vision_transformer import ToMeTransformerEncoderLayer

@MODELS.register_module
class MergeMix(BaseModel):
    def __init__(self,
                 backbone,
                 backbone_k=None,
                 head=None,
                 alpha=1.0,
                 merge_num=0,
                 source_tracking_mode='map',
                 tome_in_mix=True,
                 mask_leaked=False,
                 lam_scale=False,
                 lam_margin=-1,
                 attn_based='tome',
                 switch_off=-1.,
                 save=True,
                 save_name='MixedSamples',
                 debug=False,
                 pretrained=None,
                 pretrained_k=None,
                 init_cfg=None,
                 **kwargs):
        super(MergeMix, self).__init__(init_cfg, **kwargs)
        # basic params
        self.alpha = float(alpha)
        self.mask_leaked = bool(mask_leaked)
        self.lam_scale = bool(lam_scale)
        self.attn_based = str(attn_based)
        self.merge_num = int(merge_num)
        self.source_tracking_mode = str(source_tracking_mode)
        self.switch_off = float(switch_off)
        self.lam_margin = float(lam_margin)
        self.save = bool(save)
        self.save_name = str(save_name)
        self.ploter = PlotTensor(apply_inv=True)
        self.debug = bool(debug)
        self.tome_in_mix = bool(tome_in_mix)

        # Encoder
        if isinstance(backbone, dict):
            self.backbone = builder.build_backbone(backbone)
        else:
            raise ValueError("Ensure with the backbone.")
        if backbone_k is not None:
            self.backbone_k = builder.build_backbone(backbone_k)

        # Token Merge Settings
        self.scale_factor = 16
        assert self.source_tracking_mode in ['map', 'matrix'], "Please Choose map or matrix models."
        assert self.attn_based in ['topk', 'random', 'tome'], "Please make sure the mode of generating masks."
        self.merge_ratio = check_parse_r(len(self.backbone.layers), self.merge_num, (224 / self.scale_factor) ** 2, -0.5)
        self.backbone = self.tome_apply_patch(model=self.backbone)

        # CLS Head
        assert head is None or isinstance(head, dict)
        self.head = builder.build_head(head)

        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)


    def init_weights(self, pretrained=None, pretrained_k=None):

        # init pretrained backbone_k
        if pretrained_k is not None:
            print_log('load pre-training from: {}'.format(pretrained_k), logger='root')
            if self.backbone_k is not None:
                self.backbone_k.init_weights(pretrained=pretrained_k)

        # init trainable params
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
            load_checkpoint(self, pretrained, strict=False, logger=logging.getLogger())
            self.backbone.init_weights(pretrained=pretrained)

        # init head
        if self.head is not None:
            self.head.init_weights()


    def tome_apply_patch(self, model):

        ToMeVisionTransformer = self.make_tome_class(model.__class__)
        model.__class__ = ToMeVisionTransformer
        model.r = 0
        model._tome_info = {
            "r": model.r,
            "size": None,
            "source": None,
            "total_merge": None,
            "trace_source": True,
            "prop_attn": True,
            "class_token": model.cls_token is not None,
            "distill_token": False,
            "source_tracking_mode": self.source_tracking_mode,
            "inference": False,
        }

        if hasattr(model, "dist_token") and model.dist_token is not None:
            model._tome_info["distill_token"] = True

        for module in model.modules():
            if isinstance(module, ToMeTransformerEncoderLayer):
                module._tome_info = model._tome_info

        model.r = (self.merge_ratio, -0.5)
        model._tome_info["r"] = model.r
        model._tome_info["total_merge"] = self.merge_num

        return model


    def make_tome_class(self, transformer_class):
        class ToMeVisionTransformer(transformer_class):
            """
            Modifications:
            - Initialize r, token size, and token sources.
            """
            def forward(self, *args, **kwdargs) -> torch.Tensor:
                self._tome_info["r"] = parse_r(
                    len(self.layers), self.r, self._tome_info["total_merge"])
                self._tome_info["size"] = None
                self._tome_info["source_map"] = None
                self._tome_info["source_matrix"] = None

                return super().forward(*args, **kwdargs)
        return ToMeVisionTransformer


    @contextmanager
    def tome_disabled(self):
        """Temporarily disable ToMe merging (set r=0) for a forward pass.

        This keeps ToMe enabled for attention extraction in earlier passes, but
        turns it off during classification forwards to avoid information loss.
        """
        model = self.backbone
        prev_r = getattr(model, "r", 0)
        try:
            model.r = 0
            yield
        finally:
            model.r = prev_r

    def tome_ctx(self):
        """Return the right context manager depending on config."""
        return self.tome_disabled() if self.tome_in_mix else nullcontext()


    def lambda_scale(self, _mean, _std, _tao=1e-5):
        _lam = np.random.normal(_mean, _std, len(_std))
        _lam = (_lam - np.min(_lam)) / (np.max(_lam) - np.min(_lam) + _tao)
        if np.any(_lam < 0):
            raise ValueError("Lambda values should be >= 0.")
        if np.any(_lam > 1):
            _lam = np.clip(_lam, 0, 1)
            print_log("Warning: Lambda values were clipped to [0,1] due to floating point precision issues.", logger='root')
        return _lam


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


    def maximum_saliency(self, mask, rand_perm=True):
        '''masks.shape: batch size, 1, 14, 14 or batch size, 1, 196'''
        results = dict()

        # Scanning this matrix, and find the max one in each row
        if rand_perm:
            index = self._no_repeat_shuffle_idx(mask.size(0), ignore_failure=True).cuda()
        else:
            expanded_masks_i = mask.unsqueeze(1)  # [batch_size, 1, 196]
            expanded_masks_j = mask.unsqueeze(0)  # [1, batch_size, 196]
            binarize_masks = expanded_masks_i + expanded_masks_j
            binarize_masks = torch.where(binarize_masks > 0, torch.ones_like(binarize_masks), torch.zeros_like(binarize_masks))
            Matrix = torch.sum(binarize_masks, dim=-1)  # [batch_size, batch_size]
            index = torch.argmax(Matrix, dim=1)
        results["index"] = index

        # Redefine the mixing ratio
        lam_ = torch.sum(mask, dim=1).cpu().numpy() / mask.shape[-1]

        scale_token = int(math.sqrt(mask.shape[-1]))
        mask = mask.reshape(mask.shape[0], 1, scale_token, scale_token)
        mask_ups = F.interpolate(mask, scale_factor=self.scale_factor, mode='nearest')
        if self.mask_leaked:
            results["mask"] = torch.cat([1 - mask_ups, mask_ups], dim=1)
            results["lam"] = 1 - lam_
        else:
            results["mask"] = torch.cat([mask_ups, 1 - mask_ups], dim=1)
            results["lam"] = lam_

        return results


    def process_attention_map(self, attn, top_k, attn_based='topk'):
        batch_size = attn.shape[0]
        flat_attn = attn[:, :, 0]
        # TODO Version 1.0
        if attn_based == 'topk' or attn_based == 'tome':
            _, indices = torch.topk(flat_attn, top_k, dim=1)
        elif attn_based == 'random':
            random_indices = torch.randperm(flat_attn.size(-1)).expand(batch_size, -1)
            indices = random_indices[:, :top_k]   
        flat_attn_ = torch.zeros_like(flat_attn)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, top_k)
        flat_attn_[batch_indices, indices] = 1
        return flat_attn_


    def ranking_mixup(self, x, lam, feature):
        """ token-wise input space mixup"""
        results = dict()
        feature = torch.sum(feature, dim=1)  # multi-head -> single head
        if self.attn_based == 'tome':
            if self.source_tracking_mode == 'map':
                source = self.backbone._tome_info['source_map'][:, 1:]
                feature = token_unmerge_from_map(feature, source)
            elif self.source_tracking_mode == 'matrix':
                source = self.backbone._tome_info['source_matrix'][:, 1:, 1:]
                feature = token_unmerge(feature, source)
        elif self.attn_based in ['topk', 'random']:
            feature = feature[:, 1:, 1:]
        retain_tokens = int(lam * feature.shape[1])
        mask = self.process_attention_map(feature, retain_tokens, attn_based=self.attn_based)

        # Ensure the index and mask pairs of this mini-batch
        # 1. Input the masks which obtain from the Token Merge methods
        # 2. Bulid a matrix of mini-batch's mask and 0-1 the masks (进行与或操作来二值化)
        # 3. Ranking the masks and choose the Top-1 values, index, and new mixing ratio lambda
        mask = self.maximum_saliency(mask.float())

        if self.debug:
            results["debug_plot"] = np.zeros((3, 100))
            results["debug_plot"][0, :] = lam
            results["debug_plot"][1, :] = 1 - lam
            results["debug_plot"][2, :] = mask["lam"]
        else:
            results["debug_plot"] = None

        # mix, apply mask on x and x_
        if self.lam_margin >= lam or self.lam_margin >= 1 - lam:
            results["img_mix"] = x * lam + x[mask["index"]] * ( 1 - lam)
            mask['lam'] = [lam for i in range(x.shape[0])]
        else:
            assert mask["mask"].shape[1] == 2
            assert mask["mask"].shape[2:] == x.shape[2:], f"Invalid mask shape={mask.shape}"
            results["img_mix"] = \
                x * mask["mask"][:, 0, :, :].unsqueeze(1) + x[mask["index"], :] * mask["mask"][:, 1, :, :].unsqueeze(1)
        results["index"] = mask["index"]

        return results, mask['lam']


    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.
        Args:
            img (Tensor): Input of a batch of images, (N, C, H, W).
            gt_label (Tensor): Groundtruth onehot labels.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        Notes:
            outputs: repersentation, cls_token, attention maps if return_attn else (repersentation, cls_token)
            latent features without cls token
        """
        if isinstance(img, list):
            img = img[0]
        self.backbone._tome_info["inference"] = False

        # Step 1. Sampling a mixing ratio and encoding the samples
        lam = np.random.beta(self.alpha, self.alpha)
        outputs = self.backbone(img)
        features, _ = outputs[0], outputs[-1]

        # Sample Mixing & Backbone training
        results, lam_mix = self.ranking_mixup(img, lam, features[-1])
        # redefine mixing ratios under the merge ratio & attention socre
        if self.lam_scale:
            _lam = self.lambda_scale(_mean=self.merge_num / 196, _std=lam_mix, _tao=1e-5)
        else:
            _lam = lam_mix

        if self.switch_off < 0:
            loss_mix, loss_base = self.forward_mix(img, results["img_mix"], gt_label, results['index'], _lam)  # orl: lam_mix                
        else:
            loss_mix = self.forward_mix(img, results["img_mix"], gt_label, results['index'], _lam)  # orl: lam_mix

        losses = {
            'loss': loss_mix['loss'],
            'acc_mix': loss_mix['acc'],
        }
        if self.switch_off < 0:
            if loss_base is not None:
                losses['loss'] += loss_base['loss']
                losses['acc_base'] = loss_base['acc']

        # save img mb
        if self.save:
            self.plot_mix(results["img_mix"], img, img[results['index'], :], lam, \
                          results["index"], results["debug_plot"], "mixed sample")
                          
        return losses

    @auto_fp16(apply_to=('mixed_x', 'x'))
    def forward_mix(self, x, mixed_x, y, index, lam):
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
        loss_base = None
        loss_mix = None
        multi_lam = False
        if self.switch_off > 0:
            if np.random.random() < self.switch_off:   # vanilla mixup
                mixed_x, y_mix = mixup(x.clone(), y, 0.8, dist_mode=False)
            else:   # y_mix for mergemix
                y_mix = (y, y[index], lam)
                multi_lam = True

            if self.head is not None:
                with self.tome_ctx():
                    out_mix = self.backbone(mixed_x)[-1]
                pred_mix = self.head([out_mix])
                pred_mix[0] = pred_mix[0].type(torch.float32)  # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
                loss_mix = self.head.lam_loss(pred_mix, y_mix, multi_lam=multi_lam)
                if torch.isnan(loss_mix['loss']):
                    print_log("Warming NAN in loss_mix. Please use FP32!", logger='root')
                    loss_mix = dict(loss=None)
                return loss_mix
        else:
            assert self.head is not None, "Please make sure classification head exist."
            if np.random.random() < 2.0:   # vanilla mixup
                x, y_mix = mixup(x.clone(), y, 0.8, dist_mode=False)
            else:
                x, y_mix = cutmix(x.clone(), y, 0.2, dist_mode=False)
            with self.tome_ctx():
                out_base = self.backbone(x)[-1]
            pred_base = self.head([out_base])
            loss_base = self.head.loss(pred_base, y_mix)
            if torch.isnan(loss_base['loss']):
                print_log("Warming NAN in loss_base. Please use FP32!", logger='root')
                loss_base = dict(loss=None)

            with self.tome_ctx():
                out_mix = self.backbone(mixed_x)[-1]
            pred_mix = self.head([out_mix])
            pred_mix[0] = pred_mix[0].type(torch.float32)  # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            y_mix = (y, y[index], lam)
            loss_mix = self.head.lam_loss(pred_mix, y_mix, multi_lam=True)
            if torch.isnan(loss_mix['loss']):
                print_log("Warming NAN in loss_mix. Please use FP32!", logger='root')
                loss_mix = dict(loss=None)
            return loss_mix, loss_base

    @force_fp32(apply_to=('im_mixed', 'im_q', 'im_k',))
    def plot_mix(self, im_mixed, im_q, im_k, lam, index, debug_plot=None, name="k"):
        """ visualize mixup results, supporting 'debug' mode """
        # plot mixup results
        img = torch.cat((im_q[:4], im_k[:4], im_mixed[:4]), dim=0)
        title_name = 'lambda {}={}'.format(name, lam)
        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=4, title_name=title_name, save_name=self.save_name)
        if self.debug:
            # visualization of mini-batch index
            select_index = np.zeros((im_q.size(0), im_q.size(0)))
            for i in range(im_q.size(0)):
                select_index[i, index[i].cpu().numpy()] = 1

            plt.figure(figsize=(4, 4))
            plt.imshow(select_index, aspect='auto', vmin=0, vmax=1)
            plt.xticks(range(0, 100, 100))
            plt.title("Selecting Samples")
            # plt.show()
            _debug_path = self.save_name.split(".png")[0] + "_select_matrix.png"
            if not os.path.exists(_debug_path):
                plt.savefig(_debug_path, bbox_inches='tight')
            plt.close()

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.backbone(img)[-1:]
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]
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
        if self.tome_in_mix:
            self.backbone._tome_info["inference"] = True
        else:
            self.backbone._tome_info["inference"] = False  # False: Testing also with ToMe, True: Testing without ToMe
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
