"""
Dataset modules for deepfake detection.

Provides PyTorch Dataset classes for video and image data.
"""

from .video_dataset import VideoDataset, create_video_dataloaders
from .image_dataset import ImageDataset, create_image_dataloaders
from .data_utils import (
    create_data_splits,
    get_dataset_stats,
    balance_dataset,
    DatasetInfo
)

__all__ = [
    "VideoDataset",
    "create_video_dataloaders",
    "ImageDataset", 
    "create_image_dataloaders",
    "create_data_splits",
    "get_dataset_stats",
    "balance_dataset",
    "DatasetInfo",
]

