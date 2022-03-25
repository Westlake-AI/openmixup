import torch
import torch.nn as nn

from ..registry import HEADS
from .. import builder
from ..utils import concat_all_gather


@HEADS.register_module
class LatentPredictHead(nn.Module):
    """Head for latent feature prediction.

    This head builds a predictor, which can be any registered neck component.
    For example, BYOL and SimSiam call this head and build NonLinearNeck.
    It also implements similarity loss between two forward features.

    Args:
        predictor (dict): Config dict for module of predictor.
    """

    def __init__(self, predictor):
        super(LatentPredictHead, self).__init__()
        self.predictor = builder.build_neck(predictor)

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)

        loss = -2 * (pred_norm * target_norm).sum(dim=1).mean()
        return dict(loss=loss)


@HEADS.register_module
class LatentClsHead(nn.Module):
    """Head for latent feature classification.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes.
    """

    def __init__(self,
                 in_channels,
                 num_classes):
        super(LatentClsHead, self).__init__()
        self.predictor = nn.Linear(in_channels, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(input)
        with torch.no_grad():
            label = torch.argmax(self.predictor(target), dim=1).detach()
        loss = self.criterion(pred, label)
        return dict(loss=loss)


@HEADS.register_module
class MoCoV3Head(nn.Module):
    """Head for MoCo v3 algorithms (similar to BYOL head)

    This head builds a predictor, which can be any registered neck component.
    It also implements latent contrastive loss between two forward features.
    Part of the code is modified from:
    `<https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py>`_.

    Args:
        predictor (dict): Config dict for module of predictor.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 1.0.
    """

    def __init__(self, predictor, temperature=1.0):
        super(MoCoV3Head, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, base_out, momentum_out):
        """Forward head.

        Args:
            base_out (Tensor): NxC features from base_encoder.
            momentum_out (Tensor): NxC features from momentum_encoder.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # predictor computation
        pred = self.predictor([base_out])[0]

        # normalize
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(momentum_out, dim=1)

        # get negative samples
        target = concat_all_gather(target)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [pred, target]) / self.temperature

        # generate labels
        batch_size = logits.shape[0]
        labels = (torch.arange(batch_size, dtype=torch.long) +
                  batch_size * torch.distributed.get_rank()).cuda()

        loss = 2 * self.temperature * self.criterion(logits, labels)
        return dict(loss=loss)
