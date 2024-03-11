from __future__ import division
import torch
import math
import sys
import random
from PIL import Image
import numpy as np
import numbers
import types
import collections
import warnings

from torchvision.transforms import functional as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class Compose(object):
    """Composes several transforms together. Generalized to apply each transform
    to two images if they are supplied

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target = None):
        if target is not None:
            for t in self.transforms:
                img, target = t(img, target)
            return img, target

        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. Generalized to convert to
    tensors two images if they are supplied.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    If a target image (of shape HxW) is provided, this will also return a tensorized version of it.
    Note that no scaling will take place, and the target tensor will be returned with type long.
    Note that the output shape will not be 1xHxW but rather HxW.
    """

    def __call__(self, pic, pic2=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            pic2 (PIL Image): (optional) Second image to be converted also.

        Returns:
            Tensor(s): Converted image(s).
        """
        if pic2 is not None:
            return F.to_tensor(pic), F.to_tensor(pic2)
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
        interpolation_tg (int, optional): Desired interpolation for target. Default is
            ``PIL.Image.NEAREST``
    """

    def __init__(self, size, interpolation=Image.BILINEAR, interpolation_tg = Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.interpolation_tg = interpolation_tg

    def __call__(self, img, target=None):
        """
        Args:
            img (PIL Image): Image to be scaled.
            target (PIL Image): (optional) Target to be scaled

        Returns:
            PIL Image: Rescaled image(s).
        """
        if target is not None:
            return F.resize(img, self.size, self.interpolation), \
                   F.resize(target, self.size, self.interpolation_tg)
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): (optional) Target to be flipped

        Returns:
            PIL Image: Randomly flipped image(s).
        """
        if random.random() < self.p:
            if target is not None:
                return F.hflip(img), F.hflip(target)
            else:
                return F.hflip(img)

        if target is not None:
            return img, target
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): (optional) Target to be flipped

        Returns:
            PIL Image: Randomly flipped image(s).
        """
        if random.random() < self.p:
            if target is not None:
                return F.vflip(img), F.vflip(target)
            else:
                return F.vflip(img)

        if target is not None:
            return img, target
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, resample_tg=False, expand=False, center=None): #,fill=0
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.resample_tg = resample_tg
        self.expand = expand
        self.center = center
        # self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, target=None):
        """
            img (PIL Image): Image to be rotated.
            target (PIL Image): (optional) Target to be rotated

        Returns:
            PIL Image: Rotated image(s).
        """

        angle = self.get_params(self.degrees)

        if target is not None:
            return F.rotate(img, angle, self.resample, self.expand, self.center), \
                   F.rotate(target, angle, self.resample_tg, self.expand, self.center) # , self.fill
                   # resample = False is by default nearest, appropriate for targets
        return F.rotate(img, angle, self.resample, self.expand, self.center) #, self.fill

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string