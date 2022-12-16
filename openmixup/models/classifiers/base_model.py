import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from mmcv.runner import BaseModule, auto_fp16

import torch
import torch.nn as nn
import torch.distributed as dist

from openmixup.models.utils import Sobel


class BaseModel(BaseModule, metaclass=ABCMeta):
    """Base model class for supervised, semi- and self-supervised learning."""

    def __init__(self,
                 init_cfg=None,
                 with_sobel=False,
                 **kwargs):
        super(BaseModel, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.backbone = nn.Identity()
        self.neck = None
        self.head = nn.Identity()
        self.with_sobel = with_sobel
        self.sobel_layer = Sobel() if with_sobel else nn.Identity()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        if not self._is_init:
            super(BaseModel, self).init_weights()
        else:
            warnings.warn('This module has bee initialized, \
                please call initialize(module, init_cfg) to reinitialize it')

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        img = self.sobel_layer(img)
        x = self.backbone(img)
        return x

    @abstractmethod
    def forward_train(self, img, **kwargs):
        """
        Args:
            img ([Tensor): List of tensors. Typically these should be
                mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    def forward_test(self, img, **kwargs):
        """
        Args:
            img (Tensor): List of tensors. Typically these should be
                mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    def forward_vis(self, img, **kwargs):
        """Forward backbone features for visualization.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        feat = self.forward_backbone(img)[-1]  # tuple
        if feat.dim() == 4:
            feat = nn.functional.adaptive_avg_pool2d(feat, 1)  # NxCx1x1
        keys = ['gap']
        outs = [feat.view(feat.size(0), -1).cpu()]
        return dict(zip(keys, outs))
    
    def forward_inference(self, img, **kwargs):
        """Forward output for inference.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.

        Returns:
            tuple[Tensor]: final model outputs.
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        preds = self.head(x, post_process=True)
        return preds[0]

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, mode='train', **kwargs):
        """Forward function of model.

        Calls either forward_train, forward_test or forward_backbone function
        according to the mode.
        """
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'inference':
            return self.forward_inference(img, **kwargs)
        elif mode == 'extract':
            if len(img.size()) > 4:
                img = img[:, 0, ...].contiguous()  # contrastive data
            return self.forward_backbone(img)
        elif mode == 'vis':
            return self.forward_vis(img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer=None):
        """The iteration step during training.

        *** replacing `batch_processor` in `EpochBasedRunner` in old version ***

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs
