"""
Transform utilities for image preprocessing
"""

import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


@dataclass
class AugmentationCfg:
    """Configuration for data augmentation"""
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Module):
    """
    Resize the input image to a maximum size while maintaining aspect ratio,
    then pad to reach the exact target size.
    """

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else max
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    """Convert image to RGB format"""
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    """
    Create image transformation pipeline
    
    Args:
        image_size: Target image size
        is_train: Whether this is for training (applies augmentation)
        mean: Normalization mean values
        std: Normalization std values
        resize_longest_max: Whether to resize based on longest edge
        fill_color: Fill color for padding
        aug_cfg: Augmentation configuration
        
    Returns:
        Composed transform pipeline
    """
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
        
    normalize = Normalize(mean=mean, std=std)
    
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        
        if use_timm:
            try:
                from timm.data import create_transform  # timm can still be optional
                if isinstance(image_size, (tuple, list)):
                    assert len(image_size) >= 2
                    input_size = (3,) + image_size[-2:]
                else:
                    input_size = (3, image_size, image_size)
                # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
                aug_cfg_dict.setdefault('interpolation', 'random')
                aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
                train_transform = create_transform(
                    input_size=input_size,
                    is_training=True,
                    hflip=0.,
                    mean=mean,
                    std=std,
                    re_mode='pixel',
                    **aug_cfg_dict,
                )
            except ImportError:
                print("Warning: timm not available, falling back to basic augmentation")
                use_timm = False
        
        if not use_timm:
            train_transform = Compose([
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.get('scale', (0.9, 1.0)),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
            remaining_cfg = {k: v for k, v in aug_cfg_dict.items() if k != 'scale'}
            if remaining_cfg:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(remaining_cfg.keys())}).')
        return train_transform
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


def get_transform(image_size=224):
    """
    Get basic transform for anomaly detection
    
    Args:
        image_size: Target image size
        
    Returns:
        Tuple of (image_transform, mask_transform)
    """
    preprocess = image_transform(image_size, is_train=False, 
                               mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    target_transform = Compose([
        Resize((image_size, image_size)),
        CenterCrop(image_size),
        ToTensor()
    ])
    
    return preprocess, target_transform