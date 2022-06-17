import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
