import cv2
import numpy as np
import torch
import torch.nn as nn


class Sobel(nn.Module):
    """Sobel layer."""

    def __init__(self,
                 isotropic=False,
                 out_channels=2,
                 use_threshold=False,
                 to_grayscale=True):
        super(Sobel, self).__init__()
        self.isotropic = isotropic
        self.out_channels = out_channels
        self.use_threshold = use_threshold
        assert self.out_channels in [1, 2,]

        if to_grayscale:
            grayscale = nn.Conv2d(
                in_channels=3, out_channels=1, kernel_size=1, bias=False)
            grayscale.weight.data.fill_(1.0 / 3.0)
        else:
            grayscale = nn.Identity()
        sobel_filter = nn.Conv2d(
            in_channels=1, out_channels=self.out_channels, kernel_size=3, stride=1,
            padding=1, padding_mode='reflect', bias=False)
        w = 1.414214 if self.isotropic else 2
        sobel_kernel = np.array([[1,  0,  -1],
                                 [w,  0,  -w],
                                 [1,  0,  -1]])
        sobel_filter.weight.data[0, 0].copy_(
            torch.from_numpy(sobel_kernel).type(torch.float32))
        if self.out_channels == 2:
            sobel_filter.weight.data[1, 0].copy_(
                torch.from_numpy(sobel_kernel.T).type(torch.float32))
        self.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in self.sobel.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.sobel(x)
        if self.use_threshold:
            x_thr = torch.quantile(
                x.view(x.size(0), 1, -1), 0.80, dim=2).view(x.size(0), 1, 1, 1)
            x[x < x_thr] = 0.
        
        return x


class Laplacian(nn.Module):
    """Laplacian of Gaussian (LoG) or  layer."""

    def __init__(self,
                 mode='LoG',
                 use_threshold=False,
                 to_grayscale=True):
        super(Laplacian, self).__init__()
        self.mode = mode
        self.use_threshold = use_threshold
        assert self.mode in ['LoG', 'DoG',]
        if to_grayscale:
            grayscale = nn.Conv2d(
                in_channels=3, out_channels=1, kernel_size=1, bias=False)
            grayscale.weight.data.fill_(1.0 / 3.0)
        else:
            grayscale = nn.Identity()
        laplacian_filter = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=5, stride=1,
            padding=2, padding_mode='reflect', bias=False)
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

    @torch.no_grad()
    def forward(self, x):
        x = self.laplacian(x)
        if self.use_threshold:
            x_thr = torch.quantile(
                x.view(x.size(0), 1, -1), 0.80, dim=2).view(x.size(0), 1, 1, 1)
            x[x < x_thr] = 0.
        
        return x


class Canny(nn.Module):
    """Canny edge detection layer."""

    def __init__(self, non_max_suppression=True, edge_smooth=False, to_grayscale=True):
        super(Canny, self).__init__()

        self.non_max_suppression = non_max_suppression
        self.edge_smooth = edge_smooth
        # gray
        if to_grayscale:
            grayscale = nn.Conv2d(
                in_channels=3, out_channels=1, kernel_size=1, bias=False)
            grayscale.weight.data.fill_(1.0 / 3.0)
        else:
            grayscale = nn.Identity()
        # blur
        gaussian_1d = cv2.getGaussianKernel(7, sigma=0.8)
        gaussian_2d = gaussian_1d * gaussian_1d.T
        gaussian = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=7, stride=1,
            padding=3, padding_mode='reflect', bias=False)
        gaussian.weight.data.copy_(
            torch.from_numpy(gaussian_2d).type(torch.float32))
        # sobel
        sobel_filter = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=3, stride=1,
            padding=1, padding_mode='reflect', bias=False)
        sobel_kernel = np.array([[1,  0,  -1],
                                 [2,  0,  -2],
                                 [1,  0,  -1]])
        sobel_filter.weight.data[0, 0].copy_(
            torch.from_numpy(sobel_kernel).type(torch.float32))
        sobel_filter.weight.data[1, 0].copy_(
            torch.from_numpy(sobel_kernel.T).type(torch.float32))
        self.sobel = nn.Sequential(grayscale, gaussian, sobel_filter)

        # direction filters
        direct_kernel = np.zeros((8, 3, 3))
        for i in range(8):
            direct_kernel[i, 1, 1] = 1
            if i != 4:
                direct_kernel[i, i // 3, i % 3] = -1
            else:
                direct_kernel[i, 2, 2] = -1
        self.direct_filter = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
        self.direct_filter.weight.data.copy_(
            torch.from_numpy(direct_kernel[:, None, ...]).type(torch.float32))
        
        for p in self.sobel.parameters():
            p.requires_grad = False
        for p in self.direct_filter.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        B, _, H, W = x.size()
        grad_img = self.sobel(x)
        grad_x = grad_img[:, 0, ...].unsqueeze(1)
        grad_y = grad_img[:, 1, ...].unsqueeze(1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        grad_ori = (torch.atan2(grad_y, grad_x) * (180. / 3.14159))
        grad_ori += 180.
        grad_ori =  torch.round(grad_ori / 45.) * 45.

        if torch.__version__.split('+')[0] >= '1.7.0':
            img_threshold = torch.quantile(
                grad_mag.view(x.size(0), 1, -1), 0.85, dim=2).view(B, 1, 1, 1)
        else:
            img_threshold, _ = torch.median(grad_mag.view(x.size(0), 1, -1), dim=2)
            img_threshold = img_threshold.view(B, 1, 1, 1) * 2

        # NMS for thin edge
        if self.non_max_suppression:
            direct_img = self.direct_filter(grad_mag)

            idx_pos = (grad_ori / 45) % 8
            idx_neg = ((grad_ori / 45) + 4) % 8
            pixel_count = H * W
            pixel_range = torch.FloatTensor([range(pixel_count)]).view(1, -1).cuda()

            indices = (idx_pos.view(B, -1) * pixel_count + pixel_range).long()
            positive = direct_img.view(B, -1)[:, indices[0, :]].view(B, 1, H, W)

            indices = (idx_neg.view(B, -1).data * pixel_count + pixel_range).long()
            negative = direct_img.view(B, -1)[:, indices[0, :]].view(B, 1, H, W)

            is_max = torch.stack([positive, negative]).min(dim=0)[0] > 0.

            grad_mag[is_max==0] = 0.
        
        grad_mag[grad_mag < img_threshold] = 0.

        # binary edge and apply gaussian smooth
        if self.edge_smooth:
            img_threshold = torch.quantile(
                grad_mag.view(x.size(0), 1, -1), 0.99, dim=2).view(B, 1, 1, 1)
            grad_mag[grad_mag >= img_threshold] = 1.
            grad_mag = self.sobel[1](grad_mag)
        
        return grad_mag
