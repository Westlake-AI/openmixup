import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from ..registry import HEADS


@HEADS.register_module
class ContrastiveHead(BaseModule):
    r"""Head for contrastive learning.

    Implementation of infoNCE loss based on "A Simple Framework for
        Contrastive Learning of Visual Representations".
    <https://arxiv.org/abs/2002.05709>

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1) / self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        losses['loss'] = self.criterion(logits, labels)
        return losses


@HEADS.register_module
class HCRHead(BaseModule):
    r"""Head for contrastive learning.

    Implementation of infoNCE loss based on "Hyperspherical
        Consistency Regularization".
    <https://arxiv.org/abs/2206.00845>
    """

    def __init__(self):
        super(HCRHead, self).__init__()
        self.eps = 1e-10

    def pairwise_dist(self, x):
        x_square = x.pow(2).sum(dim=1)
        prod = x @ x.t()
        pdist = (x_square.unsqueeze(1) + \
            x_square.unsqueeze(0) - 2 * prod).clamp(min=self.eps)
        pdist[range(len(x)), range(len(x))] = 0.
        return pdist

    def forward(self, logits, embeds):
        """Forward head.

        Args:
            logits (Tensor): NxC predicted logits from the classifier.
            embeds (Tensor): Nxd projected embedding from the projector.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        logits = F.normalize(logits, dim=1)
        embeds = F.normalize(embeds, dim=1).detach()
        q1 = torch.exp( -self.pairwise_dist(logits))
        q2 = torch.exp( -self.pairwise_dist(embeds))

        losses = dict()
        losses['loss'] = -1 * (q1 * torch.log(q2 + self.eps)).mean() + \
            -1 * ((1 - q1) * torch.log((1 - q2) + self.eps)).mean()
        return losses
