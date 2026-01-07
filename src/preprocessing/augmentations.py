"""
Data augmentation pipelines for training.

Uses albumentations library for efficient augmentations.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# Try to import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

import cv2


def get_train_transforms(
    image_size: int = 224,
    horizontal_flip: float = 0.5,
    rotation_limit: int = 15,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    gaussian_noise: float = 0.1,
    jpeg_compression: Optional[Tuple[int, int]] = None
):
    """
    Get training augmentation transforms.
    
    Args:
        image_size: Target image size
        horizontal_flip: Probability of horizontal flip
        rotation_limit: Maximum rotation angle in degrees
        brightness_limit: Brightness adjustment range
        contrast_limit: Contrast adjustment range
        gaussian_noise: Probability of adding Gaussian noise
        jpeg_compression: JPEG quality range (min, max) or None
        
    Returns:
        Albumentations Compose object or custom transform
    """
    if HAS_ALBUMENTATIONS:
        transforms_list = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=horizontal_flip),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=rotation_limit,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=1
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=1
                ),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
            ], p=gaussian_noise),
        ]
        
        # Add JPEG compression augmentation (simulates real-world compression)
        if jpeg_compression:
            transforms_list.append(
                A.ImageCompression(
                    quality_lower=jpeg_compression[0],
                    quality_upper=jpeg_compression[1],
                    p=0.3
                )
            )
        
        # Normalize and convert to tensor
        transforms_list.extend([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms_list)
    else:
        # Fallback to simple transforms
        return SimpleTransform(image_size, training=True)


def get_val_transforms(image_size: int = 224):
    """
    Get validation/test transforms (minimal augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose object or custom transform
    """
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return SimpleTransform(image_size, training=False)


class SimpleTransform:
    """
    Simple transform class when albumentations is not available.
    """
    
    def __init__(self, image_size: int = 224, training: bool = False):
        """
        Initialize simple transform.
        
        Args:
            image_size: Target image size
            training: Whether this is for training (adds basic augmentation)
        """
        self.image_size = image_size
        self.training = training
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply transforms.
        
        Args:
            image: Input image (H, W, C) in RGB format
            
        Returns:
            Dictionary with 'image' key containing transformed tensor
        """
        import torch
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Training augmentations
        if self.training:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
            
            # Random brightness/contrast
            if np.random.random() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2)  # Contrast
                beta = np.random.uniform(-20, 20)           # Brightness
                image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Convert to tensor format (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        
        return {"image": image}


class VideoTransform:
    """
    Transform for video sequences.
    Applies consistent augmentation across all frames.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        training: bool = False,
        normalize: bool = True
    ):
        """
        Initialize video transform.
        
        Args:
            image_size: Target frame size
            training: Whether this is for training
            normalize: Whether to normalize pixel values
        """
        self.image_size = image_size
        self.training = training
        self.normalize = normalize
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply transforms to video frames.
        
        Args:
            frames: Input frames (N, H, W, C)
            
        Returns:
            Transformed frames as tensor (N, C, H, W)
        """
        import torch
        
        transformed = []
        
        # Determine augmentation parameters (same for all frames)
        do_flip = self.training and np.random.random() > 0.5
        alpha = 1.0 + np.random.uniform(-0.2, 0.2) if self.training else 1.0
        beta = np.random.uniform(-20, 20) if self.training else 0
        
        for frame in frames:
            # Resize
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            
            # Apply consistent augmentations
            if do_flip:
                frame = np.fliplr(frame).copy()
            
            if self.training:
                frame = np.clip(alpha * frame + beta, 0, 255).astype(np.uint8)
            
            # Normalize
            if self.normalize:
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - self.mean) / self.std
            
            # Convert to (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))
            transformed.append(frame)
        
        # Stack to (N, C, H, W)
        transformed = np.stack(transformed, axis=0)
        
        return torch.from_numpy(transformed).float()


def mixup(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply mixup augmentation.
    
    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        alpha: Beta distribution parameter
        
    Returns:
        Tuple of (mixed_images, labels_a, labels_b, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = len(images)
    index = np.random.permutation(batch_size)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    
    return mixed_images, labels_a, labels_b, lam


def cutmix(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply cutmix augmentation.
    
    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        alpha: Beta distribution parameter
        
    Returns:
        Tuple of (mixed_images, labels_a, labels_b, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size, _, h, w = images.shape
    index = np.random.permutation(batch_size)
    
    # Get random box
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply cutmix
    mixed_images = images.copy()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    
    # Adjust lambda based on actual box size
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    
    labels_a = labels
    labels_b = labels[index]
    
    return mixed_images, labels_a, labels_b, lam

