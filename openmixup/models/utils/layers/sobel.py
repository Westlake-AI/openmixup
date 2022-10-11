import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                x.view(x.size(0), 1, -1), 0.85, dim=2).view(x.size(0), 1, 1, 1)
            x[x < x_thr] = 0.
        
        return x


class HOG(nn.Module):
    """Generate hog feature for each batch images. This module is used in
    Maskfeat to generate hog feature. This code is borrowed from.
    <https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/masked.py>
    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16) -> None:
        super(HOG, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = self.get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer('gkern', gkern)

    def get_gkern(self, kernlen: int, std: int) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        gkern1d = _gaussian_fn(kernlen, std)
        gkern2d = gkern1d[:, None] * gkern1d[None, :]
        return gkern2d / gkern2d.sum()

    def _reshape(self, hog_feat: torch.Tensor, hog_h: int) -> torch.Tensor:
        b = hog_feat.shape[0]
        hog_feat = hog_feat.flatten(1, 2)
        unfold_size = hog_feat.shape[-1] // hog_h
        hog_feat = (
            hog_feat.permute(0, 2, 3, 1).unfold(
                1, unfold_size, unfold_size).unfold(
                2, unfold_size, unfold_size).flatten(1, 2).flatten(2))
        return hog_feat.permute(0, 2, 1).reshape(b, -1, hog_h, hog_h)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate hog feature for each batch images.
        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).
        Returns:
            torch.Tensor: Hog features.
        """
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out_h = int(h / self.gaussian_window)  # (14, 14)
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=torch.float,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, 'h {} gw {}'.format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = F.normalize(out, p=2, dim=2)

        return self._reshape(out, out_h)


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
                x.view(x.size(0), 1, -1), 0.85, dim=2).view(x.size(0), 1, 1, 1)
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
