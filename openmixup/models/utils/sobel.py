import torch
import torch.nn as nn


class Sobel(nn.Module):
    """Sobel layer."""

    def __init__(self, isotropic=False, out_channels=2):
        super(Sobel, self).__init__()
        self.isotropic = isotropic
        self.out_channels = out_channels
        assert self.out_channels in [1, 2,]
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        grayscale.weight.data.fill_(1.0 / 3.0)
        sobel_filter = nn.Conv2d(
            1, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        w = 1.414214 if self.isotropic else 2
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1,  0,  -1],
                               [w,  0,  -w],
                               [1,  0,  -1]]))
        if self.out_channels == 2:
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[ 1,  w,   1],
                                   [ 0,  0,   0],
                                   [-1, -w,  -1]]))
        self.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in self.sobel.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.sobel(x)


class Laplacian(nn.Module):
    """Laplacian of Gaussian (LoG) or  layer."""

    def __init__(self, mode='LoG'):
        super(Laplacian, self).__init__()
        self.mode = mode
        assert self.mode in ['LoG', 'DoG',]
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        grayscale.weight.data.fill_(1.0 / 3.0)
        laplacian_filter = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        if self.mode == 'LoG':
            laplacian_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[0,  0,   1,  0,  0,],
                                   [0,  1,   2,  1,  0,],
                                   [1,  2, -16,  2,  1,],
                                   [0,  1,   2,  1,  0,],
                                   [0,  0,   1,  0,  0,]]))
        elif self.mode == 'DoG':
            laplacian_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[ 0,  0,  -1,   0,  0],
                                   [ 0, -1,  -2,  -1,  0],
                                   [-1, -2,  16,  -2, -1],
                                   [ 0, -2,  -2,  -1,  0],
                                   [ 0,  0,  -1,   0,  0]]))
        self.laplacian = nn.Sequential(grayscale, laplacian_filter)
        for p in self.laplacian.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.laplacian(x)
