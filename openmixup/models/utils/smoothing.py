import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class Smoothing(nn.Module):
    """ Gaussian Smoothing in SuperMix
    
    Modified from `SuperMix repo <https://github.com/alldbi/SuperMix>`
    """

    def __init__(self):
        super(Smoothing, self).__init__()

    @staticmethod
    def generate_kernels(sigma=1, chennels=1):
        size_denom = 5.
        sigma = int(sigma * size_denom)
        kernel_size = sigma
        mgrid = torch.arange(kernel_size, dtype=torch.float32)
        mean = (kernel_size - 1.) / 2.
        mgrid = mgrid - mean
        mgrid = mgrid * size_denom
        kernel = 1. / (sigma * math.sqrt(2. * math.pi)) * \
                 torch.exp(-(((mgrid - 0.) / (sigma)) ** 2) * 0.5)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernelx = kernel.view(1, 1, int(kernel_size), 1).repeat(chennels, 1, 1, 1)
        kernely = kernel.view(1, 1, 1, int(kernel_size)).repeat(chennels, 1, 1, 1)

        return kernelx.cuda(), kernely.cuda(), kernel_size

    def forward(self, input, sigma):
        if sigma > 0:
            channels = input.size(1)
            kx, ky, kernel_size = self.generate_kernels(sigma, channels)
            # padd the input
            padd0 = int(kernel_size // 2)
            evenorodd = int(1 - kernel_size % 2)

            input = F.pad(
                input, (padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 'constant', 0.)
            input = F.conv2d(input, weight=kx, groups=channels)
            input = F.conv2d(input, weight=ky, groups=channels)
        return input


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Modified from `timm repo <https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598>`

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
