import torch
import torch.nn as nn


class Scale(nn.Module):
    """A learnable scale parameter by element multiplications"""

    def __init__(self, dim=1, init_value=1.0, trainable=True):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
