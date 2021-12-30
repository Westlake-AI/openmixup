import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torchvision

from torchvision import transforms
from openmixup.utils import print_log
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class AutoMixup_V1plus(nn.Module):
    """ AutoMix V0707

    Implementation of "AutoMix: Unveiling the Power of Mixup
        (https://arxiv.org/abs/2103.13027)".
    
    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        mix_block (dict): Config dict for the mixblock.
        head_mix (dict): Config dict for module of mixup classification loss.
        head_one (dict): Config dict for module of onehot classification loss.
        head_indices (tuple): Indices of the cls head.
            Default: ("head_mix_q", "head_one_q", "head_mix_k", "head_one_k")
        mask_layer (int): Number of the feature layer indix in the backbone.
        alpha (int): Beta distribution '$\beta(\alpha, \alpha)$'.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
        mask_loss (float): Loss weight for the mixup mask. Default: 0.
        lam_margin (int): Margin of lambda to stop using AutoMix to train backbone
            when lam is small. If lam > lam_margin: AutoMix; else: vanilla mixup.
            Default: -1 (or 0).
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """
    
    def __init__(self,
                 backbone,
                 mix_block,
                 head_mix,
                 head_one=None,
                 head_indices=(
                     "head_mix_q", "head_one_q",
                     "head_mix_k", "head_one_k"),
                 mask_layer=2,
                 alpha=1.0,
                 momentum=0.999,
                 mask_loss=0.,
                 lam_margin=-1,
                 save=False,
                 save_name='mixup_samples',
                 pretrained=None):
        super(AutoMixup_V1plus, self).__init__()
        # basic params
        self.alpha = alpha
        self.mask_layer = mask_layer
        self.momentum = momentum
        self.base_momentum = momentum
        self.mask_loss = mask_loss
        self.lam_margin = lam_margin
        self.save = save
        self.save_name = save_name
        assert lam_margin < 1. and mask_layer <= 4
        # mixblock
        self.mix_block = builder.build_head(mix_block)
        # backbone
        self.backbone_q = builder.build_backbone(backbone)
        self.backbone_k = builder.build_backbone(backbone)
        for param in self.backbone_k.parameters():  # stop grad k
            param.requires_grad = False
        # mixup cls head
        assert "head_mix_q" in head_indices and "head_mix_k" in head_indices
        self.head_mix_q = builder.build_head(head_mix)
        self.head_mix_k = builder.build_head(head_mix)
        for param in self.head_mix_k.parameters():  # stop grad k
            param.requires_grad = False
        # onehot cls head
        if "head_one_q" in head_indices:
            self.head_one_q = builder.build_head(head_one)
        else:
            self.head_one_q = None
        if "head_one_k" in head_indices and "head_one_q" in head_indices:
            self.head_one_k = builder.build_head(head_one)
            for param in self.head_one_k.parameters():  # stop grad k
                param.requires_grad = False
        else:
            self.head_one_k = None
        
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        # init params in q
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone_q.init_weights(pretrained=pretrained)
        if self.head_mix_q is not None:
            self.head_mix_q.init_weights(init_linear='kaiming')
        if self.head_one_q is not None:
            self.head_one_q.init_weights(init_linear='kaiming')
        
        # copy backbone param from q to k
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)
        
        # copy head one param from q to k
        if self.head_one_q is not None and self.head_one_k is not None:
            for param_one_q, param_one_k in zip(self.head_one_q.parameters(),
                                                self.head_one_k.parameters()):
                param_one_k.data.copy_(param_one_q.data)
        # copy head mix param from q to k
        if self.head_mix_q is not None and self.head_mix_k is not None:
            for param_mix_q, param_mix_k in zip(self.head_mix_q.parameters(),
                                                self.head_mix_k.parameters()):
                param_mix_k.data.copy_(param_mix_q.data)
        
        # init mixblock
        if self.mix_block is not None:
            self.mix_block.init_weights(init_linear='normal')

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the k form q, including the backbone and heads """
        # update k's backbone and cls head from q
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1 - self.momentum)
        
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

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of a batch of images, (N, C, H, W).
            gt_label (Tensor): Groundtruth onehot labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = img.size()[0]
        lam = np.random.beta(self.alpha, self.alpha, 2)
        index_q = torch.randperm(batch_size).cuda()
        index_k = torch.randperm(batch_size).cuda()

        with torch.no_grad():
            self._momentum_update()

        # auto Mixup
        indices = [index_k, index_q]
        feature = self.backbone_k(img)[0]
        mixed_x_q, mixed_x_k, loss_mask = self.pixel_mixup(img, lam, indices, feature)

        # mixed sample visualization
        if self.save:
            self.plot_mix(mixed_x_k, img, img[index_k, :], lam[0])
        
        # k: the mix block training
        loss_mix_k = self.forward_k(mixed_x_k, gt_label, index_k, lam[0])
        # q: the encoder training
        loss_one_q, loss_mix_q = self.forward_q(img, mixed_x_q, gt_label, index_q, lam[1])
        
        # loss summary
        losses = {
            'loss': loss_mix_q['loss'] + loss_mix_k['loss'],
            'acc_mix_k': loss_mix_k['acc'],
            'acc_mix_q': loss_mix_q['acc'],
        }
        if loss_one_q is not None:
            losses['loss'] += loss_one_q['loss']
            losses['acc_one_q'] = loss_one_q['acc']
        if loss_mask is not None and self.mask_loss > 0:
            losses["loss"] += loss_mask["loss"] * self.mask_loss
        return losses

    @torch.no_grad()
    def plot_mix(self, img_mixed, img, img_, lam):
        invTrans = transforms.Compose([
            transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1/0.2023, 1/0.1994, 1/0.201]),
            transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[ 1., 1., 1. ])])
        imgs = torch.cat((img[:4], img_[:4], img_mixed[:4]), dim=0)
        img_grid = torchvision.utils.make_grid(imgs, nrow=4, pad_value=0)
        imgs = np.transpose(invTrans(img_grid).detach().cpu().numpy(), (1, 2, 0))
        fig = plt.figure()
        plt.imshow(imgs)
        plt.title('lambda k: {}'.format(lam))
        if not os.path.exists(self.save_name):
            plt.savefig(self.save_name)
        plt.close()
    
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
        # onehot q
        if self.head_one_q is not None:
            out_one_q = self.backbone_q(x)[-1]
            pred_one_q = self.head_one_q([out_one_q])
            # loss
            error_one_q = (pred_one_q, y)
            loss_one_q = self.head_one_q.loss(*error_one_q)
        else:
            loss_one_q = None
        
        # mixup q
        out_mix_q = self.backbone_q(mixed_x)[-1]
        pred_mix_q = self.head_mix_q([out_mix_q])
        # mixup loss
        y_mix_q = (y, y[index], lam)
        error_mix_q = (pred_mix_q, y_mix_q)
        loss_mix_q = self.head_mix_q.loss(*error_mix_q)
        return loss_one_q, loss_mix_q

    def forward_k(self, mixed_x, y, index, lam):
        """ forward k with the mixup sample """
        # mixed_x forward
        out_mix_k = self.backbone_k(mixed_x)[-1]
        pred_mix_k = self.head_mix_k([out_mix_k])
        # k mixup loss
        y_mix_k = (y, y[index], lam)
        error_mix_k = (pred_mix_k, y_mix_k)
        loss_mix_k = self.head_mix_k.loss(*error_mix_k)
        return loss_mix_k
    
    def pixel_mixup(self, x, lam, index, feature):
        """ pixel-wise input space mixup, v07.07

        Args:
            x (Tensor): Input of a batch of images, (N, C, H, W).
            lam (List): Input list of lambda (scalar).
            index (List): Input list of shuffle index (tensor) for mixup.
            feature (Tensor): The feature map of x, (N, C, H', W').

        Returns:
            mixed_x_q, mixed_x_k: Mixup samples for q (training the backbone)
                and k (training the mixblock).
            mask_loss (Tensor): Output loss of mixup masks.
        """
        # lam info
        lam_k = lam[0]  # lam is a scalar
        lam_q = lam[1]

        # mask upsampling factor
        if x.shape[3] > 64:  # normal version of resnet
            scale_factor = 2**(2 + self.mask_layer)
        else:  # CIFAR version
            scale_factor = 2**self.mask_layer
        
        # get mixup mask
        mask_k = self.mix_block(feature, lam_k, index[0], scale_factor=scale_factor)
        mask_q = self.mix_block(feature, lam_q, index[1], scale_factor=scale_factor).clone().detach()
        # lam_margin for backbone training
        if self.lam_margin >= lam_q or self.lam_margin >= 1-lam_q:
            mask_q[:, 0, :, :] = lam_q
            mask_q[:, 1, :, :] = 1 - lam_q
        # loss of mixup mask
        if self.mask_loss > 0.:
            mask_loss = self.mix_block.loss(mask_k, lam_k)
        else:
            mask_loss = None
        
        # mix, apply mask on x and x_
        # mixed_x_k = x * (1 - mask_k) + x[index[0], :] * mask_k
        assert mask_k.shape[1] == 2 and mask_q.shape[1] == 2
        mixed_x_k = x * mask_k[:, 0, :, :].unsqueeze(1) + x[index[0], :] * mask_k[:, 1, :, :].unsqueeze(1)
        
        # mixed_x_q = x * (1 - mask_q) + x[index[1], :] * mask_q
        mixed_x_q = x * mask_q[:, 0, :, :].unsqueeze(1) + x[index[1], :] * mask_q[:, 1, :, :].unsqueeze(1)
        return mixed_x_q, mixed_x_k, mask_loss

    def forward_test(self, img, **kwargs):
        """Forward computation during testing.

        Args:
            img (Tensor): Input of a batch of images, (N, C, H, W).

        Returns:
            dict[key, Tensor]: A dictionary of head names (key) and predictions.
        """
        keys = list()  # 'acc_mix_k', 'acc_one_k', 'acc_mix_q', 'acc_one_q'
        pred = list()
        # backbone
        last_k = self.backbone_k(img)[-1]
        last_q = self.backbone_q(img)[-1]
        # head k
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
        
        out_tensors = [p[0].cpu() for p in pred]  # NxC
        return dict(zip(keys, out_tensors))

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        else:
            raise Exception('No such mode: {}'.format(mode))
