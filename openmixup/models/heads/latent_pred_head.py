import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .. import builder


@HEADS.register_module
class LatentPredictHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, size_average=True):
        super(LatentPredictHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average

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
        loss = -2 * (pred_norm * target_norm).sum()
        if self.size_average:
            loss /= input.size(0)
        return dict(loss=loss)
