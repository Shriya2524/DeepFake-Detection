"""
Tests for dataset modules.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


class TestDataUtils:
    """Tests for data utility functions."""
    
    def test_create_data_splits(self):
        """Test data splitting function."""
        from src.datasets.data_utils import create_data_splits
        
        paths = [f"video_{i}.mp4" for i in range(100)]
        labels = [0] * 50 + [1] * 50  # Balanced
        
        splits = create_data_splits(
            paths,
            labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        train_paths, train_labels = splits['train']
        val_paths, val_labels = splits['val']
        test_paths, test_labels = splits['test']
        
        # Check approximate sizes
        assert len(train_paths) == 70
        assert len(val_paths) == 15
        assert len(test_paths) == 15
        
        # Check no overlap
        all_paths = set(train_paths) | set(val_paths) | set(test_paths)
        assert len(all_paths) == 100
    
    def test_balance_dataset_undersample(self):
        """Test dataset undersampling."""
        from src.datasets.data_utils import balance_dataset
        
        paths = [f"file_{i}" for i in range(100)]
        labels = [0] * 30 + [1] * 70  # Imbalanced
        
        bal_paths, bal_labels = balance_dataset(
            paths,
            labels,
            method="undersample",
            seed=42
        )
        
        # Should have 30 of each class
        assert bal_labels.count(0) == 30
        assert bal_labels.count(1) == 30
    
    def test_balance_dataset_oversample(self):
        """Test dataset oversampling."""
        from src.datasets.data_utils import balance_dataset
        
        paths = [f"file_{i}" for i in range(100)]
        labels = [0] * 30 + [1] * 70
        
        bal_paths, bal_labels = balance_dataset(
            paths,
            labels,
            method="oversample",
            seed=42
        )
        
        # Should have 70 of each class
        assert bal_labels.count(0) == 70
        assert bal_labels.count(1) == 70
    
    def test_get_dataset_stats(self):
        """Test computing dataset statistics."""
        from src.datasets.data_utils import get_dataset_stats
        
        paths = [f"file_{i}" for i in range(100)]
        labels = [0] * 40 + [1] * 60
        
        stats = get_dataset_stats(paths, labels, name="test_dataset")
        
        assert stats.name == "test_dataset"
        assert stats.total_samples == 100
        assert stats.num_real == 40
        assert stats.num_fake == 60


class TestImageDataset:
    """Tests for ImageDataset class."""
    
    def test_dataset_init(self):
        """Test dataset initialization with dummy paths."""
        from src.datasets.image_dataset import ImageDataset
        
        # Create dummy paths (files don't need to exist for init)
        paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        labels = [0, 1, 0]
        
        dataset = ImageDataset(
            image_paths=paths,
            labels=labels,
            extract_faces=False,
            use_multichannel=False,
            transform=None
        )
        
        assert len(dataset) == 3
    
    def test_get_num_channels(self):
        """Test channel counting."""
        from src.datasets.image_dataset import ImageDataset
        
        dataset = ImageDataset(
            image_paths=["img.jpg"],
            labels=[0],
            use_multichannel=True,
            use_ycbcr=True,
            use_frequency=True,
            use_edges=True
        )
        
        # RGB(3) + CbCr(2) + DCT(1) + Edges(3) = 9
        assert dataset.get_num_channels() == 9


class TestVideoDataset:
    """Tests for VideoDataset class."""
    
    def test_dataset_init(self):
        """Test dataset initialization."""
        from src.datasets.video_dataset import VideoDataset
        
        paths = ["video1.mp4", "video2.mp4"]
        labels = [0, 1]
        
        dataset = VideoDataset(
            video_paths=paths,
            labels=labels,
            target_fps=10,
            max_frames=16,
            extract_faces=False,
            compute_flow=False
        )
        
        assert len(dataset) == 2
    
    def test_pad_sequence(self):
        """Test sequence padding."""
        from src.datasets.video_dataset import VideoDataset
        
        dataset = VideoDataset(
            video_paths=["v.mp4"],
            labels=[0],
            max_frames=32
        )
        
        # Test padding
        short_tensor = torch.randn(10, 3, 224, 224)
        padded, mask = dataset._pad_sequence(short_tensor, 32)
        
        assert padded.shape[0] == 32
        assert mask.sum() == 10
        
        # Test truncation
        long_tensor = torch.randn(50, 3, 224, 224)
        truncated, mask = dataset._pad_sequence(long_tensor, 32)
        
        assert truncated.shape[0] == 32
        assert mask.sum() == 32


class TestDataLoaderCreation:
    """Tests for dataloader creation functions."""
    
    def test_create_image_dataloaders(self):
        """Test image dataloader creation."""
        from src.datasets.image_dataset import create_image_dataloaders
        
        train_paths = [f"train_{i}.jpg" for i in range(10)]
        train_labels = [0] * 5 + [1] * 5
        val_paths = [f"val_{i}.jpg" for i in range(4)]
        val_labels = [0] * 2 + [1] * 2
        
        # This would fail if files don't exist, but tests the function signature
        # In real tests, you'd use fixtures or mock
        try:
            dataloaders = create_image_dataloaders(
                train_paths=train_paths,
                train_labels=train_labels,
                val_paths=val_paths,
                val_labels=val_labels,
                batch_size=2,
                num_workers=0
            )
        except FileNotFoundError:
            # Expected since files don't exist
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

