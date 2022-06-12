import math
import random
import inspect
import warnings
from typing import Sequence, Tuple
import numpy as np
from numbers import Number
from PIL import Image, ImageFilter

import torch
from torchvision import transforms as _transforms
import torchvision.transforms.functional as F

from openmixup.utils import build_from_cfg
from ..registry import PIPELINES
from ..utils import to_numpy, to_tensor


# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur', 'AutoAugment', 'RandAugment', 'RandomErasing']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@PIPELINES.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomErasing_numpy(object):
    """Randomly selects a rectangle region in an image and erase pixels.
        using Numpy backend (return to PIL)
        *** modified from MMClassification, used before normalization! ***
    
    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 0.4
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:

            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]

        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.

    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_

        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:

        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(self,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=0.4,
                 aspect_range=(3 / 10, 10 / 3),
                 mode='const',
                 fill_color=(128, 128, 128),
                 fill_std=None):
        assert isinstance(erase_prob, float) and 0. <= erase_prob <= 1.
        assert isinstance(min_area_ratio, float) and 0. <= min_area_ratio <= 1.
        assert isinstance(max_area_ratio, float) and 0. <= max_area_ratio <= 1.
        assert min_area_ratio <= max_area_ratio, \
            'min_area_ratio should be smaller than max_area_ratio'
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert isinstance(aspect_range, Sequence) and len(aspect_range) == 2 \
            and all(isinstance(x, float) for x in aspect_range), \
            'aspect_range should be a float or Sequence with two float.'
        assert all(x > 0 for x in aspect_range), \
            'aspect_range should be positive.'
        assert aspect_range[0] <= aspect_range[1], \
            'In aspect_range (min, max), min should be smaller than max.'
        assert mode in ['const', 'rand']
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert isinstance(fill_color, Sequence) and len(fill_color) == 3 \
            and all(isinstance(x, Number) for x in fill_color), \
            'fill_color should be a float or Sequence with three int.'
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert isinstance(fill_std, Sequence) and len(fill_std) == 3 \
                and all(isinstance(x, Number) for x in fill_std), \
                'fill_std should be a float or Sequence with three int.'

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

    def _fill_pixels(self, img, top, left, h, w):
        if self.mode == 'const':
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top:top + h, left:left + w] = patch
        return img

    def __call__(self, img):
        if np.random.rand() > self.erase_prob:
            return img
        img = np.array(img)
        img_h, img_w = img.shape[:2]

        # convert to log aspect to ensure equal probability of aspect ratio
        log_aspect_range = np.log(
            np.array(self.aspect_range, dtype=np.float32))
        aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
        area = img_h * img_w
        area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

        h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
        w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
        top = np.random.randint(0, img_h - h) if img_h > h else 0
        left = np.random.randint(0, img_w - w) if img_w > w else 0
        img = self._fill_pixels(img, top, left, h, w)

        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(erase_prob={self.erase_prob}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_area_ratio={self.max_area_ratio}, '
        repr_str += f'aspect_range={self.aspect_range}, '
        repr_str += f'mode={self.mode}, '
        repr_str += f'fill_color={self.fill_color}, '
        repr_str += f'fill_std={self.fill_std})'
        return repr_str


@PIPELINES.register_module
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        *** modified from timm, used after normalization! ***

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
        erase_prob (float): Probability that the Random Erasing operation will be
            performed. Default: 0.5.
        min_area_ratio (float): Minimum erased area / input image area. Default: 0.02.
        max_area_ratio (float): Maximum erased area / input image area. Default: 1/3.
        min_aspect (float): Minimum aspect ratio of erased area. Default: 3/10.
        max_aspect (float): Maximum aspect ratio of erased area. Default: 10/3.
        mode (str): pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color in [0, 255]
            'pixel' - erase block is per-pixel random (normal) color in [0, 255]
        min_count (float): Minimum number of erasing blocks per image. Default: 1.
        max_count (float): Maximum number of erasing blocks per image, area per box is
            scaled by count. per-image count is randomly chosen between 1 and this value.
    """

    def __init__(self,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=1/3,
                 min_aspect=3/10,
                 max_aspect=10/3,
                 mode='const',
                 min_count=1,
                 max_count=None,
                 num_splits=0,
                 device='cuda'):
        assert isinstance(erase_prob, float) and 0. <= erase_prob <= 1.
        assert isinstance(min_area_ratio, float) and 0. <= min_area_ratio <= 1.
        assert isinstance(max_area_ratio, float) and 0. <= max_area_ratio <= 1.
        assert min_area_ratio <= max_area_ratio, \
            'min_area_ratio should be smaller than max_area_ratio'
        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device

    def _get_pixels(self, patch_size, dtype=torch.float32, device='cuda'):
        # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
        # paths, flip the order so normal is run on CPU if this becomes a problem
        # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
        if self.per_pixel:
            return torch.empty(patch_size, dtype=dtype, device=device).normal_()
        elif self.rand_color:
            return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
        else:
            return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.erase_prob:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for _ in range(10):
                target_area = random.uniform(
                    self.min_area_ratio, self.max_area_ratio) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = \
                        self._get_pixels((chan, h, w), dtype=dtype, device=self.device)
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

    def __repr__(self):
        # NOTE simplified state for repr
        repr_str = self.__class__.__name__
        repr_str += f'(erase_prob={self.erase_prob}, mode={self.mode}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_area_ratio={self.max_area_ratio}, '
        repr_str += f'count=({self.min_count}, {self.max_count}))'
        return repr_str


@PIPELINES.register_module()
class BEiTMaskGenerator(object):
    """Generate mask for the image in BEiT.

    This module is borrowed from
    https://github.com/microsoft/unilm/tree/master/beit

    Args:
        input_size (int): The size of input image.
        num_masking_patches (int): The number of patches to be masked.
        min_num_patches (int): The minimum number of patches to be masked
            in the process of generating mask. Defaults to 4.
        max_num_patches (int, optional): The maximum number of patches to be
            masked in the process of generating mask. Defaults to None.
        min_aspect (float): The minimum aspect ratio of mask blocks. Defaults
            to 0.3.
        min_aspect (float, optional): The minimum aspect ratio of mask blocks.
            Defaults to None.
        mask_only (bool): Whether to only return the mask. Defaults to False.
    """

    def __init__(self,
                 input_size=(14, 14),
                 num_masking_patches=75,
                 min_num_patches=16,
                 max_num_patches=None,
                 min_aspect=0.3,
                 max_aspect=None,
                 mask_only=False,
                ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None \
            else max_num_patches
        self.mask_only = mask_only

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches,
                                         max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top:top + h, left:left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(
        self, img: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count != self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            mask_count += delta

        if self.mask_only:
            return mask
        else:
            return img[0], img[1], mask

    def __repr__(self) -> None:
        repr_str = 'Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)' % (
            self.height, self.width, self.min_num_patches,
            self.max_num_patches, self.num_masking_patches,
            self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str


@PIPELINES.register_module()
class BlockwiseMaskGenerator(object):
    """Generate random block for the image.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
        mask_color (str): Filling color of the MIM mask in {'mean', 'zero'}.
            Defaults to 'zero'.
    """

    def __init__(self,
                 input_size=192,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.6,
                 mask_only=False,
                 mask_color='zero',
                ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_only = mask_only
        self.mask_color = mask_color
        assert self.mask_color in ['mean', 'zero', 'rand',]
        if self.mask_color != 'zero':
            assert mask_only == False

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img=None) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.from_numpy(mask)  # [H, W]

        if self.mask_color == 'mean':
            if isinstance(img, Image.Image):
                img = np.array(img)
                mask_ = to_numpy(mask).reshape((self.rand_size * self.scale, -1, 1))
                mask_ = mask_.repeat(
                    self.model_patch_size, axis=0).repeat(self.model_patch_size, axis=1)
                mean = img.reshape(-1, img.shape[2]).mean(axis=0)
                img = np.where(mask_ == 1, img, mean)
                img = Image.fromarray(img.astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                mask_ = to_tensor(mask)
                mask_ = mask_.repeat_interleave(self.model_patch_size, 0).repeat_interleave(
                    self.model_patch_size, 1).contiguous()
                img = img.clone()
                mean = img.mean(dim=[1,2])
                for i in range(img.size(0)):
                    img[i, mask_ == 1] = mean[i]

        if self.mask_only:
            return mask
        else:
            return img, mask


@PIPELINES.register_module()
class RandomResizedCropWithTwoCrop(object):
    """Crop the given PIL Image to random size and aspect ratio with random
    interpolation into two crops.

    This module is borrowed from
    https://github.com/microsoft/unilm/tree/master/beit.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size. This is popularly used
    to train the Inception networks. This module first crops the image and
    resizes the crop to two different sizes.

    Args:
        size (Union[tuple, int]): Expected output size of each edge of the
            first image.
        second_size (Union[tuple, int], optional): Expected output size of each
            edge of the second image.
        scale (tuple[float, float]): Range of size of the origin size cropped.
            Defaults to (0.08, 1.0).
        ratio (tuple[float, float]): Range of aspect ratio of the origin aspect
            ratio cropped. Defaults to (3./4., 4./3.).
        interpolation (str): The interpolation for the first image. Defaults
            to ``bilinear``.
        second_interpolation (str): The interpolation for the second image.
            Defaults to ``lanczos``.
    """

    interpolation_dict = {
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }

    def __init__(self,
                 size=224,
                 second_size=None,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear',
                 second_interpolation='lanczos') -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn('range should be of kind (min, max)')

        if interpolation == 'random':
            self.interpolation = (Image.BILINEAR, Image.BICUBIC)
        else:
            self.interpolation = self.interpolation_dict.get(
                interpolation, Image.BILINEAR)
        self.second_interpolation = self.interpolation_dict.get(
            second_interpolation, Image.BILINEAR)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: np.ndarray, scale: tuple, ratio: tuple) -> Sequence[int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect
                ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation), \
                F.resized_crop(img, i, j, h, w, self.second_size,
                               self.second_interpolation)


# custom transforms
@PIPELINES.register_module
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise).
    
    Args:
        alphastd (float, optional): The parameter for Lighting.
            Defaults to 0.1.
    """

    _IMAGENET_PCA = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203],
            ])
    }

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), \
            "Expect torch.Tensor, got {}".format(type(img))
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class GaussianBlur(object):
    """GaussianBlur augmentation refers to `SimCLR.

    <https://arxiv.org/abs/2002.05709>`_.

    Args:
        sigma_min (float): The minimum parameter of Gaussian kernel std.
        sigma_max (float): The maximum parameter of Gaussian kernel std.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, sigma_min, sigma_max, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sigma_min = {self.sigma_min}, '
        repr_str += f'sigma_max = {self.sigma_max}, '
        repr_str += f'prob = {self.prob}'
        return repr_str


@PIPELINES.register_module
class Solarization(object):
    """Solarization augmentation refers to `BYOL.

    <https://arxiv.org/abs/2006.07733>`_.

    Args:
        threshold (float, optional): The solarization threshold.
            Defaults to 128.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, threshold=128, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0, 1], got {p} instead.'
        self.threshold = threshold
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 -img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'threshold = {self.threshold}, '
        repr_str += f'prob = {self.prob}'
        return repr_str


@PIPELINES.register_module
class PlaceCrop(object):
    """Crops the given Image at the particular index list in Co-Tuning,
        https://proceedings.neurips.cc//paper/2020/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf.
        Usually used as a test time augmentation.
    
    Args:
        size (tuple or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size).
        start (tuple or list or int): The start coordinate for CenterCrop. If
            start is a sequence or list, it will be uniformly sampled each time.
    """

    def __init__(self, size, start):
        self.size = size
        self.start = start
        if isinstance(size, int):
            self.size = (int(size), int(size))

    def __call__(self, img):
        start_x = self.start
        start_y = self.start
        if isinstance(self.start, list) or isinstance(self.start, tuple):
            start_x = self.start[np.random.randint(len(self.start))]
            start_y = self.start[np.random.randint(len(self.start))]
        th, tw = self.size
        return img.crop((start_x, start_y, start_x + tw, start_y + th))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
