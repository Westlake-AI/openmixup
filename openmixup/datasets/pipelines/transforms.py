import math
import random
import inspect
import warnings
from numbers import Number
from typing import Sequence, Tuple

import mmcv
import numpy as np
from PIL import Image, ImageFilter, ImageOps

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
class RandomChoiceTrans(object):
    """Apply single transformation randomly picked from a list.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float or list): Probability.
    """

    def __init__(self, transforms, p=None):
        if p is not None:
            assert not isinstance(p, Sequence), "Argument p should be a sequence"
            assert len(p) == len(transforms)
        self.trans = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.p = p

    def __call__(self, *args):
        t = random.choices(self.trans, weights=self.p)[0]
        return t(*args)

    def __repr__(self) -> str:
        return f"{super().__repr__()}(p={self.p})"


@PIPELINES.register_module()
class CenterCropForEfficientNet(object):
    r"""Center crop the image.

    Args:
        size (int | tuple): Expected size after cropping with the format
            of (h, w).
        efficientnet_style (bool): Whether to use efficientnet style center
            crop. Defaults to False.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet style is True. Defaults to
            32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Only valid if
            ``efficientnet_style`` is True. Defaults to 'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Only valid if efficientnet style is True.
            Defaults to `cv2`.


    Notes:
        - If the image is smaller than the crop size, return the original
          image.
        - If efficientnet_style is set to False, the pipeline would be a simple
          center crop using the size.
        - If efficientnet_style is set to True, the pipeline will be to first
          to perform the center crop with the ``crop_size_`` as:

        .. math::
            \text{size} = \frac{\text{size}}{\text{size} +
            \text{crop_padding}} \times \text{short_edge}

        And then the pipeline resizes the img to the input crop size.
    """

    def __init__(self,
                 size,
                 efficientnet_style=False,
                 crop_padding=32,
                 interpolation='bilinear',
                 backend='cv2'):
        if efficientnet_style:
            assert isinstance(size, int)
            assert crop_padding >= 0
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                     'lanczos')
            if backend not in ['cv2', 'pillow']:
                raise ValueError(
                    f'backend: {backend} is not supported for '
                    'resize. Supported backends are "cv2", "pillow"')
        else:
            assert isinstance(size, int) or (isinstance(size, tuple)
                                                  and len(size) == 2)
        if isinstance(size, int):
            crop_size = (size, size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.efficientnet_style = efficientnet_style
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, img):
        """
        Args:
            img (PIL): Image to be cropped.
        """
        img = np.array(img)
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        # img.shape has length 2 for grayscale, length 3 for color
        img_height, img_width = img.shape[:2]

        # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L118 # noqa
        if self.efficientnet_style:
            img_short = min(img_height, img_width)
            crop_height = crop_height / (crop_height +
                                            self.crop_padding) * img_short
            crop_width = crop_width / (crop_width +
                                        self.crop_padding) * img_short

        y1 = max(0, int(round((img_height - crop_height) / 2.)))
        x1 = max(0, int(round((img_width - crop_width) / 2.)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1

        # crop the image
        img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))

        if self.efficientnet_style:
            img = mmcv.imresize(
                img,
                tuple(self.crop_size[::-1]),
                interpolation=self.interpolation,
                backend=self.backend)

        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@PIPELINES.register_module()
class RandomResizedCropForEfficient(object):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Args:
        size (sequence | int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Defaults to (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        efficientnet_style (bool): Whether to use efficientnet style Random
            ResizedCrop. Defaults to False.
        min_covered (Number): Minimum ratio of the cropped area to the original
             area. Only valid if efficientnet_style is true. Defaults to 0.1.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet_style is true.
            Defaults to 32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Defaults to `cv2`.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 max_attempts=10,
                 efficientnet_style=False,
                 min_covered=0.1,
                 crop_padding=32,
                 interpolation='bilinear',
                 backend='cv2'):
        if efficientnet_style:
            assert isinstance(size, int)
            self.size = (size, size)
            assert crop_padding >= 0
        else:
            if isinstance(size, (tuple, list)):
                self.size = size
            else:
                self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received scale {scale} and rato {ratio}.')
        assert min_covered >= 0, 'min_covered should be no less than 0.'
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.efficientnet_style = efficientnet_style
        self.min_covered = min_covered
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio, max_attempts=10):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

    # https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py # noqa
    @staticmethod
    def get_params_efficientnet_style(img,
                                      size,
                                      scale,
                                      ratio,
                                      max_attempts=10,
                                      min_covered=0.1,
                                      crop_padding=32):
        """Get parameters for ``crop`` for a random sized crop in efficientnet
        style.

        Args:
            img (ndarray): Image to be cropped.
            size (sequence): Desired output size of the crop.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.
            min_covered (Number): Minimum ratio of the cropped area to the
                original area. Only valid if efficientnet_style is true.
                Defaults to 0.1.
            crop_padding (int): The crop padding parameter in efficientnet
                style center crop. Defaults to 32.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height, width = img.shape[:2]
        area = height * width
        min_target_area = scale[0] * area
        max_target_area = scale[1] * area

        for _ in range(max_attempts):
            aspect_ratio = random.uniform(*ratio)
            min_target_height = int(
                round(math.sqrt(min_target_area / aspect_ratio)))
            max_target_height = int(
                round(math.sqrt(max_target_area / aspect_ratio)))

            if max_target_height * aspect_ratio > width:
                max_target_height = int((width + 0.5 - 1e-7) / aspect_ratio)
                if max_target_height * aspect_ratio > width:
                    max_target_height -= 1

            max_target_height = min(max_target_height, height)
            min_target_height = min(max_target_height, min_target_height)

            # slightly differs from tf implementation
            target_height = int(
                round(random.uniform(min_target_height, max_target_height)))
            target_width = int(round(target_height * aspect_ratio))
            target_area = target_height * target_width

            # slight differs from tf. In tf, if target_area > max_target_area,
            # area will be recalculated
            if (target_area < min_target_area or target_area > max_target_area
                    or target_width > width or target_height > height
                    or target_area < min_covered * area):
                continue

            ymin = random.randint(0, height - target_height)
            xmin = random.randint(0, width - target_width)
            ymax = ymin + target_height - 1
            xmax = xmin + target_width - 1

            return ymin, xmin, ymax, xmax

        # Fallback to central crop
        img_short = min(height, width)
        crop_size = size[0] / (size[0] + crop_padding) * img_short

        ymin = max(0, int(round((height - crop_size) / 2.)))
        xmin = max(0, int(round((width - crop_size) / 2.)))
        ymax = min(height, ymin + crop_size) - 1
        xmax = min(width, xmin + crop_size) - 1

        return ymin, xmin, ymax, xmax

    def __call__(self, img):
        """
        Args:
            img (PIL): Image to be cropped.
        """
        img = np.array(img)
        if self.efficientnet_style:
            get_params_func = self.get_params_efficientnet_style
            get_params_args = dict(
                img=img,
                size=self.size,
                scale=self.scale,
                ratio=self.ratio,
                max_attempts=self.max_attempts,
                min_covered=self.min_covered,
                crop_padding=self.crop_padding)
        else:
            get_params_func = self.get_params
            get_params_args = dict(
                img=img,
                scale=self.scale,
                ratio=self.ratio,
                max_attempts=self.max_attempts)
        ymin, xmin, ymax, xmax = get_params_func(**get_params_args)
        img = mmcv.imcrop(img, bboxes=np.array([xmin, ymin, xmax, ymax]))
        img = mmcv.imresize(
            img,
            tuple(self.size[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(size={self.size}'
        repr_str += f', scale={tuple(round(s, 4) for s in self.scale)}'
        repr_str += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', min_covered={self.min_covered}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop_mmcls(object):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Args:
        size (sequence | int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Defaults to (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        efficientnet_style (bool): Whether to use efficientnet style Random
            ResizedCrop. Defaults to False.
        min_covered (Number): Minimum ratio of the cropped area to the original
             area. Only valid if efficientnet_style is true. Defaults to 0.1.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet_style is true.
            Defaults to 32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Defaults to `cv2`.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 max_attempts=10,
                 efficientnet_style=False,
                 min_covered=0.1,
                 crop_padding=32,
                 interpolation='bilinear',
                 backend='cv2'):
        if efficientnet_style:
            assert isinstance(size, int)
            self.size = (size, size)
            assert crop_padding >= 0
        else:
            if isinstance(size, (tuple, list)):
                self.size = size
            else:
                self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received scale {scale} and rato {ratio}.')
        assert min_covered >= 0, 'min_covered should be no less than 0.'
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.efficientnet_style = efficientnet_style
        self.min_covered = min_covered
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio, max_attempts=10):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

    # https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py # noqa
    @staticmethod
    def get_params_efficientnet_style(img,
                                      size,
                                      scale,
                                      ratio,
                                      max_attempts=10,
                                      min_covered=0.1,
                                      crop_padding=32):
        """Get parameters for ``crop`` for a random sized crop in efficientnet
        style.

        Args:
            img (ndarray): Image to be cropped.
            size (sequence): Desired output size of the crop.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.
            min_covered (Number): Minimum ratio of the cropped area to the
                original area. Only valid if efficientnet_style is true.
                Defaults to 0.1.
            crop_padding (int): The crop padding parameter in efficientnet
                style center crop. Defaults to 32.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height, width = img.shape[:2]
        area = height * width
        min_target_area = scale[0] * area
        max_target_area = scale[1] * area

        for _ in range(max_attempts):
            aspect_ratio = random.uniform(*ratio)
            min_target_height = int(
                round(math.sqrt(min_target_area / aspect_ratio)))
            max_target_height = int(
                round(math.sqrt(max_target_area / aspect_ratio)))

            if max_target_height * aspect_ratio > width:
                max_target_height = int((width + 0.5 - 1e-7) / aspect_ratio)
                if max_target_height * aspect_ratio > width:
                    max_target_height -= 1

            max_target_height = min(max_target_height, height)
            min_target_height = min(max_target_height, min_target_height)

            # slightly differs from tf implementation
            target_height = int(
                round(random.uniform(min_target_height, max_target_height)))
            target_width = int(round(target_height * aspect_ratio))
            target_area = target_height * target_width

            # slight differs from tf. In tf, if target_area > max_target_area,
            # area will be recalculated
            if (target_area < min_target_area or target_area > max_target_area
                    or target_width > width or target_height > height
                    or target_area < min_covered * area):
                continue

            ymin = random.randint(0, height - target_height)
            xmin = random.randint(0, width - target_width)
            ymax = ymin + target_height - 1
            xmax = xmin + target_width - 1

            return ymin, xmin, ymax, xmax

        # Fallback to central crop
        img_short = min(height, width)
        crop_size = size[0] / (size[0] + crop_padding) * img_short

        ymin = max(0, int(round((height - crop_size) / 2.)))
        xmin = max(0, int(round((width - crop_size) / 2.)))
        ymax = min(height, ymin + crop_size) - 1
        xmax = min(width, xmin + crop_size) - 1

        return ymin, xmin, ymax, xmax

    def __call__(self, img):
        if self.efficientnet_style:
            get_params_func = self.get_params_efficientnet_style
            get_params_args = dict(
                img=img,
                size=self.size,
                scale=self.scale,
                ratio=self.ratio,
                max_attempts=self.max_attempts,
                min_covered=self.min_covered,
                crop_padding=self.crop_padding)
        else:
            get_params_func = self.get_params
            get_params_args = dict(
                img=img,
                scale=self.scale,
                ratio=self.ratio,
                max_attempts=self.max_attempts)
        ymin, xmin, ymax, xmax = get_params_func(**get_params_args)
        img = mmcv.imcrop(img, bboxes=np.array([xmin, ymin, xmax, ymax]))
        img = mmcv.imresize(
            img,
            tuple(self.size[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(size={self.size}'
        repr_str += f', scale={tuple(round(s, 4) for s in self.scale)}'
        repr_str += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', min_covered={self.min_covered}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip_mmcls(object):
    """Flip the image randomly.

    Flip the image randomly based on flip probaility and flip direction.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_prob=0.5, direction='horizontal'):
        assert 0 <= flip_prob <= 1
        assert direction in ['horizontal', 'vertical']
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self, img):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.flip_prob else False
        # results['flip'] = flip
        # results['flip_direction'] = self.direction
        # if results['flip']:
        #     # flip image
        #     for key in results.get('img_fields', ['img']):
        #         results[key] = mmcv.imflip(
        #             results[key], direction=results['flip_direction'])
        if flip:
            img = mmcv.imflip(img, self.direction)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


@PIPELINES.register_module()
class Resize_mmcls(object):
    """Resize images.

    Args:
        size (int | tuple): Images scales for resizing (h, w).
            When size is int, the default behavior is to resize an image
            to (size, size). When size is tuple and the second value is -1,
            the image will be resized according to adaptive_side. For example,
            when size is 224, the image is resized to 224x224. When size is
            (224, -1) and adaptive_size is "short", the short side is resized
            to 224 and the other side is computed based on the short side,
            maintaining the aspect ratio.
        interpolation (str): Interpolation method. For "cv2" backend, accepted
            values are "nearest", "bilinear", "bicubic", "area", "lanczos". For
            "pillow" backend, accepted values are "nearest", "bilinear",
            "bicubic", "box", "lanczos", "hamming".
            More details can be found in `mmcv.image.geometric`.
        adaptive_side(str): Adaptive resize policy, accepted values are
            "short", "long", "height", "width". Default to "short".
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Default: `cv2`.
    """

    def __init__(self,
                 size,
                 interpolation='bilinear',
                 adaptive_side='short',
                 backend='cv2'):
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        assert adaptive_side in {'short', 'long', 'height', 'width'}

        self.adaptive_side = adaptive_side
        self.adaptive_resize = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.adaptive_resize = True
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')
        if backend == 'cv2':
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                     'lanczos')
        else:
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'box',
                                     'lanczos', 'hamming')
        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, img):
        ignore_resize = False
        if self.adaptive_resize:
            h, w = img.shape[:2]
            target_size = self.size[0]

            condition_ignore_resize = {
                'short': min(h, w) == target_size,
                'long': max(h, w) == target_size,
                'height': h == target_size,
                'width': w == target_size
            }

            if condition_ignore_resize[self.adaptive_side]:
                ignore_resize = True
            elif any([
                    self.adaptive_side == 'short' and w < h,
                    self.adaptive_side == 'long' and w > h,
                    self.adaptive_side == 'width',
            ]):
                width = target_size
                height = int(target_size * h / w)
            else:
                height = target_size
                width = int(target_size * w / h)
        else:
            height, width = self.size
        if not ignore_resize:
            img = mmcv.imresize(
                img,
                size=(width, height),
                interpolation=self.interpolation,
                return_scale=False,
                backend=self.backend)
        return img

    def __call__(self, img):
        img = self._resize_img(img)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class CenterCrop_mmcls(object):
    r"""Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping with the format
            of (h, w).
        efficientnet_style (bool): Whether to use efficientnet style center
            crop. Defaults to False.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet style is True. Defaults to
            32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Only valid if
            ``efficientnet_style`` is True. Defaults to 'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Only valid if efficientnet style is True.
            Defaults to `cv2`.


    Notes:
        - If the image is smaller than the crop size, return the original
          image.
        - If efficientnet_style is set to False, the pipeline would be a simple
          center crop using the crop_size.
        - If efficientnet_style is set to True, the pipeline will be to first
          to perform the center crop with the ``crop_size_`` as:

        .. math::
            \text{crop_size_} = \frac{\text{crop_size}}{\text{crop_size} +
            \text{crop_padding}} \times \text{short_edge}

        And then the pipeline resizes the img to the input crop size.
    """

    def __init__(self,
                 crop_size,
                 efficientnet_style=False,
                 crop_padding=32,
                 interpolation='bilinear',
                 backend='cv2'):
        if efficientnet_style:
            assert isinstance(crop_size, int)
            assert crop_padding >= 0
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                     'lanczos')
            if backend not in ['cv2', 'pillow']:
                raise ValueError(
                    f'backend: {backend} is not supported for '
                    'resize. Supported backends are "cv2", "pillow"')
        else:
            assert isinstance(crop_size, int) or (isinstance(crop_size, tuple)
                                                  and len(crop_size) == 2)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.efficientnet_style = efficientnet_style
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, img):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        # img.shape has length 2 for grayscale, length 3 for color
        img_height, img_width = img.shape[:2]

        # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L118 # noqa
        if self.efficientnet_style:
            img_short = min(img_height, img_width)
            crop_height = crop_height / (crop_height +
                                            self.crop_padding) * img_short
            crop_width = crop_width / (crop_width +
                                        self.crop_padding) * img_short

        y1 = max(0, int(round((img_height - crop_height) / 2.)))
        x1 = max(0, int(round((img_width - crop_width) / 2.)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1

        # crop the image
        img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))

        if self.efficientnet_style:
            img = mmcv.imresize(
                img,
                tuple(self.crop_size[::-1]),
                interpolation=self.interpolation,
                backend=self.backend)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@PIPELINES.register_module
class RandomErasing_mmcls(object):
    """Randomly selects a rectangle region in an image and erase pixels.
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
        # img = np.array(img)
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

        # return Image.fromarray(img.astype(np.uint8))
        return img

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


@PIPELINES.register_module()
class Normalize_mmcls(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, img):
        img = mmcv.imnormalize(img, self.mean, self.std,
                               self.to_rgb)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ImageToTensor(object):

    def __init__(self):
        pass

    def __call__(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = to_tensor(img.transpose(2, 0, 1))
        return img

    def __repr__(self):
        return self.__class__.__name__


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
                 max_area_ratio=0.4,
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


@PIPELINES.register_module
class Lighting_mmcls(object):
    """Adjust images lighting using AlexNet-style PCA jitter.

    Args:
        eigval (Sequence[float]): the eigenvalue of the convariance matrix
            of pixel values, respectively.
        eigvec (list[list]): the eigenvector of the convariance matrix of
            pixel values, respectively.
        alphastd (float): The standard deviation for distribution of alpha.
            Defaults to 0.1.
    """

    def __init__(self,
                 eigval: Sequence[float],
                 eigvec: Sequence[float],
                 alphastd: float = 0.1,
                ):
        assert isinstance(eigval, Sequence), \
            f'eigval must be Sequence, got {type(eigval)} instead.'
        assert isinstance(eigvec, Sequence), \
            f'eigvec must be Sequence, got {type(eigvec)} instead.'
        for vec in eigvec:
            assert isinstance(vec, Sequence) and len(vec) == len(eigvec[0]), \
                'eigvec must contains lists with equal length.'
        assert isinstance(alphastd, float), 'alphastd should be of type ' \
            f'float or int, got {type(alphastd)} instead.'

        self.eigval = np.array(eigval)
        self.eigvec = np.array(eigvec)
        self.alphastd = alphastd

    def __call__(self, img):
        if self.alphastd <= 0.:
            return img
        img = np.array(img)
        img = mmcv.adjust_lighting(
            img,
            self.eigval,
            self.eigvec,
            alphastd=self.alphastd,
            to_rgb=False)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(eigval={self.eigval.tolist()}, '
        repr_str += f'eigvec={self.eigvec.tolist()}, '
        repr_str += f'alphastd={self.alphastd}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise).
    
    Args:
        eigval (Sequence[float]): the eigenvalue of the convariance matrix
            of pixel values, respectively.
        eigvec (list[list]): the eigenvector of the convariance matrix of
            pixel values, respectively.
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

    def __init__(self,
                 eigval=None,
                 eigvec=None,
                 alphastd=0.1):

        if isinstance(eigval, Sequence):
            self.eigval = torch.from_numpy(np.array(eigval))
        else:
            self.eigval = self._IMAGENET_PCA['eigval']
        if isinstance(eigvec, Sequence):
            self.eigvec = torch.from_numpy(np.array(eigvec))
        else:
            self.eigvec = self._IMAGENET_PCA['eigvec']
        assert isinstance(alphastd, float), 'alphastd should be of type ' \
            f'float or int, got {type(alphastd)} instead.'

        self.alphastd = alphastd

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
        repr_str += f'(eigval={self.eigval.numpy().tolist()}, '
        repr_str += f'eigvec={self.eigvec.numpy().tolist()}, '
        repr_str += f'alphastd={self.alphastd}, '
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

    def __init__(self, sigma_min=0.1, sigma_max=2.0, p=0.5):
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

    def __init__(self, threshold=128, p=0.5, backend='pillow'):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0, 1], got {p} instead.'
        self.threshold = threshold
        self.prob = p
        self.backend = backend

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        if self.backend == 'pillow':
            return ImageOps.solarize(img)
        else:
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


@PIPELINES.register_module
class ToHalf(object):
    """ Convert to torch.Tensor (torch.fp16) """

    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = img.to(torch.half)
        else:
            img = img.astype(np.float16)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
