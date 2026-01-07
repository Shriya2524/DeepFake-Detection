"""
Video dataset for spatiotemporal deepfake detection.

Loads videos, extracts frames, computes optical flow, and
prepares data for the spatiotemporal model.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..preprocessing.video_reader import VideoReader
from ..preprocessing.face_extraction import FaceExtractor
from ..preprocessing.optical_flow import OpticalFlowExtractor
from ..preprocessing.augmentations import VideoTransform
from ..utils.logger import get_logger


logger = get_logger(__name__)


class VideoDataset(Dataset):
    """
    PyTorch Dataset for video-based deepfake detection.
    
    Loads videos, extracts and processes frames, optionally computes
    optical flow, and returns sequences ready for the spatiotemporal model.
    
    Example:
        dataset = VideoDataset(
            video_paths=["video1.mp4", "video2.mp4"],
            labels=[0, 1],
            target_fps=10,
            max_frames=32
        )
        frames, flow, label = dataset[0]
    """
    
    def __init__(
        self,
        video_paths: List[Union[str, Path]],
        labels: List[int],
        target_fps: int = 10,
        max_frames: int = 32,
        min_frames: int = 8,
        image_size: int = 224,
        extract_faces: bool = True,
        compute_flow: bool = True,
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        training: bool = True
    ):
        """
        Initialize video dataset.
        
        Args:
            video_paths: List of paths to video files
            labels: List of labels (0=real, 1=fake)
            target_fps: Target FPS for frame sampling
            max_frames: Maximum number of frames per video
            min_frames: Minimum frames required
            image_size: Target frame size
            extract_faces: Whether to extract faces from frames
            compute_flow: Whether to compute optical flow
            transform: Optional transform to apply to frames
            cache_dir: Optional directory for caching processed videos
            training: Whether this is for training (affects augmentation)
        """
        assert len(video_paths) == len(labels), "Paths and labels must have same length"
        
        self.video_paths = [Path(p) for p in video_paths]
        self.labels = labels
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.image_size = image_size
        self.extract_faces = extract_faces
        self.compute_flow = compute_flow
        self.training = training
        
        # Initialize processors
        self.video_reader = VideoReader(
            target_fps=target_fps,
            max_frames=max_frames,
            min_frames=min_frames
        )
        
        if extract_faces:
            self.face_extractor = FaceExtractor(
                target_size=(image_size, image_size),
                padding=0.2
            )
        else:
            self.face_extractor = None
        
        if compute_flow:
            self.flow_extractor = OpticalFlowExtractor()
        else:
            self.flow_extractor = None
        
        # Set up transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = VideoTransform(
                image_size=image_size,
                training=training
            )
        
        # Cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate paths
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that all video paths exist."""
        invalid = [p for p in self.video_paths if not p.exists()]
        if invalid:
            logger.warning(f"Found {len(invalid)} invalid video paths")
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with:
            - 'frames': Tensor of shape (N, C, H, W)
            - 'flow': Tensor of shape (N-1, 2, H, W) if compute_flow
            - 'label': Label tensor
            - 'mask': Valid frame mask
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Try to load from cache
            cached = self._load_from_cache(idx)
            if cached is not None:
                return cached
            
            # Read video frames
            frames = self.video_reader.read(video_path)
            
            # Extract faces if enabled
            if self.extract_faces and self.face_extractor:
                face_frames = []
                for frame in frames:
                    face = self.face_extractor.extract_primary_face(frame)
                    if face is not None:
                        face_frames.append(face)
                    else:
                        # Use center crop as fallback
                        h, w = frame.shape[:2]
                        size = min(h, w)
                        start_h = (h - size) // 2
                        start_w = (w - size) // 2
                        crop = frame[start_h:start_h+size, start_w:start_w+size]
                        crop = np.array(
                            __import__('cv2').resize(crop, (self.image_size, self.image_size))
                        )
                        face_frames.append(crop)
                
                frames = np.array(face_frames)
            
            # Compute optical flow if enabled
            flow = None
            if self.compute_flow and self.flow_extractor and len(frames) > 1:
                flow = self.flow_extractor.compute_sequence(frames)
            
            # Apply transforms
            if self.transform:
                frames = self.transform(frames)
            else:
                # Manual conversion
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            
            # Process optical flow
            if flow is not None:
                flow = torch.from_numpy(flow).permute(0, 3, 1, 2).float()
                # Resize flow to match frame size
                if flow.shape[-2:] != frames.shape[-2:]:
                    flow = torch.nn.functional.interpolate(
                        flow,
                        size=frames.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
            else:
                # Create dummy flow
                flow = torch.zeros(
                    max(1, len(frames) - 1), 2, self.image_size, self.image_size
                )
            
            # Pad or truncate to max_frames
            frames, mask = self._pad_sequence(frames, self.max_frames)
            flow, _ = self._pad_sequence(flow, self.max_frames - 1)
            
            result = {
                'frames': frames,
                'flow': flow,
                'label': torch.tensor(label, dtype=torch.long),
                'mask': mask,
                'path': str(video_path)
            }
            
            # Cache result
            self._save_to_cache(idx, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            # Return dummy data
            return self._get_dummy_sample(label)
    
    def _pad_sequence(
        self,
        tensor: torch.Tensor,
        target_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or truncate sequence to target length.
        
        Args:
            tensor: Input tensor (N, ...)
            target_length: Target sequence length
            
        Returns:
            Tuple of (padded_tensor, valid_mask)
        """
        current_length = len(tensor)
        
        if current_length >= target_length:
            # Truncate
            return tensor[:target_length], torch.ones(target_length, dtype=torch.bool)
        else:
            # Pad
            pad_length = target_length - current_length
            pad_shape = (pad_length,) + tensor.shape[1:]
            padding = torch.zeros(pad_shape, dtype=tensor.dtype)
            
            padded = torch.cat([tensor, padding], dim=0)
            mask = torch.cat([
                torch.ones(current_length, dtype=torch.bool),
                torch.zeros(pad_length, dtype=torch.bool)
            ])
            
            return padded, mask
    
    def _get_dummy_sample(self, label: int) -> Dict[str, torch.Tensor]:
        """Get dummy sample for error cases."""
        return {
            'frames': torch.zeros(self.max_frames, 3, self.image_size, self.image_size),
            'flow': torch.zeros(self.max_frames - 1, 2, self.image_size, self.image_size),
            'label': torch.tensor(label, dtype=torch.long),
            'mask': torch.zeros(self.max_frames, dtype=torch.bool),
            'path': ''
        }
    
    def _get_cache_path(self, idx: int) -> Optional[Path]:
        """Get cache file path for an index."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"sample_{idx}.pt"
    
    def _load_from_cache(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load sample from cache if available."""
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, idx: int, data: Dict[str, torch.Tensor]) -> None:
        """Save sample to cache."""
        cache_path = self._get_cache_path(idx)
        if cache_path:
            try:
                # Don't cache the path string
                cache_data = {k: v for k, v in data.items() if k != 'path'}
                torch.save(cache_data, cache_path)
            except Exception:
                pass


def create_video_dataloaders(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
    batch_size: int = 8,
    val_batch_size: int = 16,
    num_workers: int = 4,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_paths: Training video paths
        train_labels: Training labels
        val_paths: Validation video paths
        val_labels: Validation labels
        test_paths: Optional test video paths
        test_labels: Optional test labels
        batch_size: Training batch size
        val_batch_size: Validation/test batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for VideoDataset
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' dataloaders
    """
    # Training dataset
    train_dataset = VideoDataset(
        video_paths=train_paths,
        labels=train_labels,
        training=True,
        **dataset_kwargs
    )
    
    # Validation dataset
    val_dataset = VideoDataset(
        video_paths=val_paths,
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
        test_dataset = VideoDataset(
            video_paths=test_paths,
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

