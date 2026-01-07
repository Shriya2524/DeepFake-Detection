"""
Image dataset for GAN fingerprint detection.

Loads images/frames and prepares multi-channel inputs for
the GAN fingerprint detection model.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..preprocessing.face_extraction import FaceExtractor
from ..preprocessing.frequency_analysis import (
    FrequencyAnalyzer,
    compute_edge_features,
    convert_color_spaces
)
from ..preprocessing.augmentations import get_train_transforms, get_val_transforms
from ..utils.logger import get_logger


logger = get_logger(__name__)


class ImageDataset(Dataset):
    """
    PyTorch Dataset for image-based GAN fingerprint detection.
    
    Creates multi-channel inputs combining RGB, color space channels,
    frequency features, and edge features.
    
    Example:
        dataset = ImageDataset(
            image_paths=["img1.jpg", "img2.jpg"],
            labels=[0, 1],
            use_multichannel=True
        )
        image, label = dataset[0]
    """
    
    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: List[int],
        image_size: int = 224,
        extract_faces: bool = True,
        use_multichannel: bool = True,
        use_ycbcr: bool = True,
        use_frequency: bool = True,
        use_edges: bool = True,
        transform: Optional[Callable] = None,
        training: bool = True
    ):
        """
        Initialize image dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of labels (0=real, 1=fake)
            image_size: Target image size
            extract_faces: Whether to extract faces
            use_multichannel: Whether to create multi-channel input
            use_ycbcr: Include YCbCr color channels
            use_frequency: Include frequency domain features
            use_edges: Include edge detection features
            transform: Optional transform (overrides default)
            training: Whether this is for training
        """
        assert len(image_paths) == len(labels), "Paths and labels must have same length"
        
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.image_size = image_size
        self.extract_faces = extract_faces
        self.use_multichannel = use_multichannel
        self.use_ycbcr = use_ycbcr
        self.use_frequency = use_frequency
        self.use_edges = use_edges
        self.training = training
        
        # Initialize face extractor
        if extract_faces:
            self.face_extractor = FaceExtractor(
                target_size=(image_size, image_size),
                padding=0.2
            )
        else:
            self.face_extractor = None
        
        # Initialize frequency analyzer
        if use_frequency:
            self.freq_analyzer = FrequencyAnalyzer(dct_size=image_size)
        else:
            self.freq_analyzer = None
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif training:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)
        
        # Validate paths
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that image paths exist."""
        invalid = [p for p in self.image_paths if not p.exists()]
        if invalid:
            logger.warning(f"Found {len(invalid)} invalid image paths")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an image sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with:
            - 'image': Tensor of shape (C, H, W)
            - 'label': Label tensor
            - 'multichannel' (optional): Multi-channel tensor
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract face if enabled
            if self.extract_faces and self.face_extractor:
                face = self.face_extractor.extract_primary_face(image)
                if face is not None:
                    image = face
                else:
                    # Center crop fallback
                    image = self._center_crop(image)
            
            # Resize
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            result = {'label': torch.tensor(label, dtype=torch.long)}
            
            # Create multi-channel input if enabled
            if self.use_multichannel:
                multichannel = self._create_multichannel(image)
                result['multichannel'] = multichannel
            
            # Apply standard transform for RGB image
            if self.transform:
                transformed = self.transform(image=image)
                result['image'] = transformed['image']
            else:
                # Manual normalization
                image = image.astype(np.float32) / 255.0
                image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                result['image'] = torch.from_numpy(image.transpose(2, 0, 1)).float()
            
            result['path'] = str(image_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return self._get_dummy_sample(label)
    
    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        """Center crop image to square."""
        h, w = image.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        return image[start_h:start_h+size, start_w:start_w+size]
    
    def _create_multichannel(self, image: np.ndarray) -> torch.Tensor:
        """
        Create multi-channel input for GAN fingerprinting.
        
        Combines: RGB (3) + CbCr (2) + DCT (1) + Sobel (2) + Laplacian (1) = 9 channels
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Multi-channel tensor (C, H, W)
        """
        channels = []
        
        # Normalize RGB to [0, 1]
        rgb = image.astype(np.float32) / 255.0
        channels.append(rgb)  # 3 channels
        
        # YCbCr color space
        if self.use_ycbcr:
            ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            cb = ycbcr[:, :, 1:2].astype(np.float32) / 255.0
            cr = ycbcr[:, :, 2:3].astype(np.float32) / 255.0
            channels.extend([cb, cr])  # 2 channels
        
        # Frequency domain
        if self.use_frequency and self.freq_analyzer:
            high_freq = self.freq_analyzer.extract_high_frequency(image)
            high_freq = high_freq[:, :, np.newaxis]
            channels.append(high_freq)  # 1 channel
        
        # Edge features
        if self.use_edges:
            sobel_x, sobel_y, laplacian = compute_edge_features(image)
            channels.extend([
                sobel_x[:, :, np.newaxis],
                sobel_y[:, :, np.newaxis],
                laplacian[:, :, np.newaxis]
            ])  # 3 channels
        
        # Concatenate all channels
        multichannel = np.concatenate(channels, axis=2)
        
        # Convert to tensor (C, H, W)
        multichannel = torch.from_numpy(multichannel.transpose(2, 0, 1)).float()
        
        return multichannel
    
    def _get_dummy_sample(self, label: int) -> Dict[str, torch.Tensor]:
        """Get dummy sample for error cases."""
        num_channels = 3  # RGB
        if self.use_multichannel:
            num_channels = 3 + (2 if self.use_ycbcr else 0) + (1 if self.use_frequency else 0) + (3 if self.use_edges else 0)
        
        result = {
            'image': torch.zeros(3, self.image_size, self.image_size),
            'label': torch.tensor(label, dtype=torch.long),
            'path': ''
        }
        
        if self.use_multichannel:
            result['multichannel'] = torch.zeros(num_channels, self.image_size, self.image_size)
        
        return result
    
    def get_num_channels(self) -> int:
        """Get the number of channels in multi-channel input."""
        num_channels = 3  # RGB
        if self.use_ycbcr:
            num_channels += 2
        if self.use_frequency:
            num_channels += 1
        if self.use_edges:
            num_channels += 3
        return num_channels


class FrameDataset(ImageDataset):
    """
    Dataset for extracting frames from videos and treating them as images.
    
    Useful for training the GAN fingerprint model on video frames.
    """
    
    def __init__(
        self,
        video_paths: List[Union[str, Path]],
        labels: List[int],
        frames_per_video: int = 10,
        **kwargs
    ):
        """
        Initialize frame dataset.
        
        Args:
            video_paths: List of video file paths
            labels: List of labels for each video
            frames_per_video: Number of frames to extract per video
            **kwargs: Additional arguments for ImageDataset
        """
        # Extract frames from videos
        image_paths = []
        image_labels = []
        
        from ..preprocessing.video_reader import VideoReader
        reader = VideoReader(max_frames=frames_per_video)
        
        for video_path, label in zip(video_paths, labels):
            try:
                frames = reader.read(video_path)
                # Save frames temporarily or keep in memory
                # For simplicity, we'll track video path + frame index
                for i in range(len(frames)):
                    image_paths.append((video_path, i))
                    image_labels.append(label)
            except Exception as e:
                logger.warning(f"Could not process video {video_path}: {e}")
        
        self.frame_info = list(zip(image_paths, image_labels))
        self.video_reader = reader
        
        # Initialize parent with dummy paths (we override __getitem__)
        super().__init__(
            image_paths=[Path("dummy")] * len(self.frame_info),
            labels=image_labels,
            **kwargs
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a frame sample."""
        (video_path, frame_idx), label = self.frame_info[idx]
        
        try:
            frames = self.video_reader.read(video_path)
            if frame_idx < len(frames):
                image = frames[frame_idx]
            else:
                image = frames[-1]
            
            # Process as regular image
            if self.extract_faces and self.face_extractor:
                face = self.face_extractor.extract_primary_face(image)
                if face is not None:
                    image = face
                else:
                    image = self._center_crop(image)
            
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            result = {'label': torch.tensor(label, dtype=torch.long)}
            
            if self.use_multichannel:
                result['multichannel'] = self._create_multichannel(image)
            
            if self.transform:
                transformed = self.transform(image=image)
                result['image'] = transformed['image']
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading frame from {video_path}: {e}")
            return self._get_dummy_sample(label)


def create_image_dataloaders(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
    batch_size: int = 32,
    val_batch_size: int = 64,
    num_workers: int = 4,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders for images.
    
    Args:
        train_paths: Training image paths
        train_labels: Training labels
        val_paths: Validation image paths
        val_labels: Validation labels
        test_paths: Optional test image paths
        test_labels: Optional test labels
        batch_size: Training batch size
        val_batch_size: Validation/test batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for ImageDataset
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' dataloaders
    """
    # Training dataset
    train_dataset = ImageDataset(
        image_paths=train_paths,
        labels=train_labels,
        training=True,
        **dataset_kwargs
    )
    
    # Validation dataset
    val_dataset = ImageDataset(
        image_paths=val_paths,
        labels=val_labels,
        training=False,
        **dataset_kwargs
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    # Test dataset if provided
    if test_paths and test_labels:
        test_dataset = ImageDataset(
            image_paths=test_paths,
            labels=test_labels,
            training=False,
            **dataset_kwargs
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return dataloaders

