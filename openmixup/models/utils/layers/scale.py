import torch
import torch.nn as nn


class Scale(nn.Module):
<<<<<<< HEAD
    """A learnable scale parameter."""

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
=======
    """A learnable scale parameter by element multiplications"""

    def __init__(self, dim=1, init_value=1.0, trainable=True):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)

    def forward(self, x):
        return x * self.scale
