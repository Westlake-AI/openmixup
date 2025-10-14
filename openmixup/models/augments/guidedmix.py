import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
try:
    import sys
    from pairing import onecycle_cover
except: pass
from copy import deepcopy

DTYPE = np.float32
INF = np.inf

def onecycle_cover(distance_matrix):
    row = 0
    col = 0
    first_row = 0
    idx = 0
    sorted_indices = np.zeros(distance_matrix.shape[0], dtype=np.int32)
    np.fill_diagonal(distance_matrix, -INF)
    max_idx = np.argmax(distance_matrix)
    row = max_idx // distance_matrix.shape[0]
    col = max_idx % distance_matrix.shape[0]
    first_row = deepcopy(row)
    sorted_indices[row] = col
    distance_matrix[:, row] = -INF
    while idx < distance_matrix.shape[0] - 2:
        idx += 1
        row = col
        col = np.argmax(distance_matrix[row])
        sorted_indices[row] = col
        distance_matrix[:, row] = -INF
    sorted_indices[col] = first_row

    return sorted_indices


@torch.no_grad()
def guidedmix(img,
              gt_label,
              alpha=1.0,
              lam=None,
              dist_mode=False,
              features=None,
              guided_type='ap',
              condition='greedy',
              distance_metric='l2',
              size=(7,7),
              sigma=(3,3),
              return_mask=False,
              **kwargs):
    r""" GuidedMixup augmentation

    "GuidedMixup: An Efficient Mixup Strategy Guided by Saliency Maps
    Based Image Classification (https://arxiv.org/abs/2306.16612)". In AAAI, 2023.
        https://github.com/3neutronstar/GuidedMixup
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        features (tensor): Feature maps for attentive regions.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """

    def cosine_similarity(a, b):

        dot = a.matmul(b.t())
        norm =a.norm(dim=1, keepdim=True).matmul(b.norm(dim=1, keepdim=True).t())

        return dot / norm

    def distance_function(a, b=None, distance_metric='jsd'):
        ''' pytorch distance 
        input:
        - a: (batch_size1 N, n_features)
        - b: (batch_size2 M, n_features)
        output: NxM matrix'''

        if b is None:
            b = a
        if distance_metric=='cosine':
            distance = 1 - cosine_similarity(a.view(a.shape[0],-1), b.view(b.shape[0],-1))
        elif distance_metric=='cosine_abs':
            distance = 1 - cosine_similarity(a.view(a.shape[0],-1), b.view(b.shape[0],-1)).abs()
        elif distance_metric =='l1':
            ra = a.view(a.shape[0],-1).unsqueeze(1)
            rb = b.view(b.shape[0],-1).unsqueeze(0)
            distance = (ra-rb).abs().sum(dim=-1).view(a.shape[0],b.shape[0])
        elif distance_metric =='l2':
            ra = a.view(a.shape[0],-1).unsqueeze(1)
            rb = b.view(b.shape[0],-1).unsqueeze(0)
            distance = ((ra-rb).norm(dim=-1)).view(a.shape[0],b.shape[0])
        else:
            raise NotImplementedError
        return distance

    assert isinstance(condition, str) and isinstance(distance_metric, str)
    if lam is None:
        lam = np.random.beta(alpha, alpha)

    if not dist_mode:
        # normal mixup process
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:  # [N, C, H, W]
            img_ = img[rand_index]
        else:
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # Notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        y_a = gt_label
        y_b = gt_label[rand_index]

        features = F.gaussian_blur(features, size, sigma)
        features /= (features).sum(dim=[-1, -2], keepdim=True)
        features_ = features[rand_index]

        if condition == 'greedy':
            x = distance_function(features.detach(), features_.detach(), distance_metric).cpu().numpy()
            rand_index = onecycle_cover(x)
        else:
            raise ValueError("Please check the condition setting in the greedy.")

        norm_features = torch.div(features, (features + features_).detach())
        lam = norm_features.mean(dim=[-1, -2]).unsqueeze(-1)
        lam_ = 1 - lam
        mask = torch.stack([norm_features] * 3, dim=1)

    img = mask * img + (1 - mask) * img_
    if return_mask:
        img = (img, mask)

    return img, (y_a, y_b, lam, lam_)


class Identity(object):
    def __init__(self):
        pass

    def __call__(self,img):
        return img

def series_filter(values, kernel_size=3):
    """
    Filter a time series. Practically, calculated mean value inside kernel size.
    As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.
    :param values:
    :param kernel_size:
    :return: The list of filtered average
    """
    filter_values = torch.cumsum(values,dim=[2,3], dtype=torch.float)

    filter_values[kernel_size:] = filter_values[kernel_size:] - filter_values[:-kernel_size]
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values

class SpectralResidual(object):
    def __init__(self,
                 blur=7,
                 sigma=3,
                 kernel_size=3,
                 device='cuda'
                 ):

        self.kernel_size = int(kernel_size)
        self.boxfilter = nn.Conv2d(in_channels=1,
                                   out_channels=1,
                                   kernel_size=kernel_size,
                                   bias=False,
                                   padding=int((kernel_size-1)/2),
                                   padding_mode='replicate')
        self.boxfilter.weight= nn.Parameter(torch.ones((1, 1, self.kernel_size, self.kernel_size), dtype=torch.float) / float(self.kernel_size * self.kernel_size), requires_grad=False)
        self.boxfilter=self.boxfilter.to(device)
        self.to_gray = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])
        if blur==0:
            self.blur=Identity()
        else:
            self.blur=transforms.GaussianBlur((blur,blur),(sigma,sigma))
        self.eps=1e-10
        self.resize=transforms.Resize((128,128))

    def transform_saliency_map(self, values):
        """
        Transform a time-series into spectral residual, which is method in computer vision.
        For example, See https://github.com/uoip/SpectralResidualSaliency.
        :param values: a list or numpy array of float values.
        :return: silency map and spectral residual
        """
        values=self.to_gray(values)
        if values.shape[2] > 128:
            values=self.resize(values)

        freq = torch.fft.fft2(values)
        mag = (freq.real ** 2 + freq.imag ** 2+self.eps).sqrt()
        spectral_residual = (mag.log() - self.boxfilter(mag.log())).exp()

        freq.real = freq.real * spectral_residual / mag
        freq.imag = freq.imag * spectral_residual / mag

        saliency_map = torch.fft.ifft2(freq)
        saliency_map.squeeze_(1)
        return saliency_map

    def transform_spectral_residual(self, values):
        with torch.no_grad():
            saliency_map = self.transform_saliency_map(values)
            spectral_residual = (saliency_map.real ** 2 + saliency_map.imag ** 2).sqrt()
            spectral_residual=self.blur(spectral_residual)
            if values.shape[2] > 128:
                spectral_residual=F.resize(spectral_residual,size=(values.shape[2],values.shape[3]))
            spectral_residual.squeeze_(1)
        
        # del for using sum-to-1 normalize
        # spectral_residual=(spectral_residual-spectral_residual.min())/(spectral_residual.max()-spectral_residual.min())

        return spectral_residual