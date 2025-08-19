"""
Transform utilities for image preprocessing
"""

import torchvision.transforms as transforms


def normalize(pred, max_value=None, min_value=None):
    """
    Normalize prediction values to [0, 1] range
    
    Args:
        pred: Input tensor to normalize
        max_value: Maximum value for normalization (if None, use pred.max())
        min_value: Minimum value for normalization (if None, use pred.min())
    
    Returns:
        Normalized tensor
    """
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def get_transform(args):
    """
    Get transform compatible with D2D interface
    
    Args:
        args: Arguments object with resolution attribute
    
    Returns:
        Tuple of (preprocess_transform, target_transform)
    """
    resolution = getattr(args, 'resolution', 224)
    
    # Basic preprocessing for images
    preprocess = transforms.Compose([
        transforms.Resize(size=(resolution, resolution), 
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Transform for ground truth masks
    target_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor()
    ])
    
    return preprocess, target_transform


def get_transform_simple(image_size=224):
    """
    Get basic image transforms for anomaly detection
    
    Args:
        image_size: Target image size
    
    Returns:
        Tuple of (preprocess_transform, target_transform)
    """
    # Basic preprocessing for images
    preprocess = transforms.Compose([
        transforms.Resize(size=(image_size, image_size), 
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Transform for ground truth masks
    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    return preprocess, target_transform