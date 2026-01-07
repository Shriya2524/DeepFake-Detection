"""
Dataset utilities for data splitting, balancing, and analysis.

Provides functions for creating train/val/test splits and
handling various dataset formats.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class DatasetInfo:
    """Container for dataset statistics and metadata."""
    name: str
    total_samples: int
    num_real: int
    num_fake: int
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"DatasetInfo(name='{self.name}', total={self.total_samples}, "
            f"real={self.num_real}, fake={self.num_fake}, "
            f"train={self.train_samples}, val={self.val_samples}, test={self.test_samples})"
        )


def create_data_splits(
    file_paths: List[Union[str, Path]],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        file_paths: List of file paths
        labels: List of corresponding labels
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        stratify: Whether to maintain class distribution in splits
        seed: Random seed for reproducibility
        max_samples: Optional maximum number of samples to use
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        (paths, labels) tuple
    """
    assert len(file_paths) == len(labels), "Paths and labels must have same length"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    np.random.seed(seed)
    
    # Convert to lists
    paths = [str(p) for p in file_paths]
    labels = list(labels)
    
    # Limit samples if specified
    if max_samples and max_samples < len(paths):
        indices = np.random.choice(len(paths), max_samples, replace=False)
        paths = [paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    n_samples = len(paths)
    
    if stratify:
        # Stratified split
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        train_indices, val_indices, test_indices = [], [], []
        
        for label, indices in class_indices.items():
            np.random.shuffle(indices)
            n = len(indices)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_indices.extend(indices[:train_end])
            val_indices.extend(indices[train_end:val_end])
            test_indices.extend(indices[val_end:])
        
        # Shuffle each split
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
    else:
        # Random split
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_indices = indices[:train_end].tolist()
        val_indices = indices[train_end:val_end].tolist()
        test_indices = indices[val_end:].tolist()
    
    # Create splits
    splits = {
        'train': (
            [paths[i] for i in train_indices],
            [labels[i] for i in train_indices]
        ),
        'val': (
            [paths[i] for i in val_indices],
            [labels[i] for i in val_indices]
        ),
        'test': (
            [paths[i] for i in test_indices],
            [labels[i] for i in test_indices]
        )
    }
    
    logger.info(
        f"Data split: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )
    
    return splits


def get_dataset_stats(
    paths: List[str],
    labels: List[int],
    name: str = "dataset"
) -> DatasetInfo:
    """
    Compute dataset statistics.
    
    Args:
        paths: List of file paths
        labels: List of labels
        name: Dataset name
        
    Returns:
        DatasetInfo object with statistics
    """
    label_counts = Counter(labels)
    
    return DatasetInfo(
        name=name,
        total_samples=len(paths),
        num_real=label_counts.get(0, 0),
        num_fake=label_counts.get(1, 0),
        class_distribution={str(k): v for k, v in label_counts.items()}
    )


def balance_dataset(
    paths: List[str],
    labels: List[int],
    method: str = "undersample",
    seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Balance dataset by undersampling or oversampling.
    
    Args:
        paths: List of file paths
        labels: List of labels
        method: Balancing method ("undersample" or "oversample")
        seed: Random seed
        
    Returns:
        Tuple of (balanced_paths, balanced_labels)
    """
    np.random.seed(seed)
    
    # Group by label
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    class_counts = {k: len(v) for k, v in class_indices.items()}
    
    if method == "undersample":
        target_count = min(class_counts.values())
    else:  # oversample
        target_count = max(class_counts.values())
    
    balanced_indices = []
    
    for label, indices in class_indices.items():
        n = len(indices)
        
        if n < target_count:
            # Oversample
            extra = np.random.choice(indices, target_count - n, replace=True)
            balanced_indices.extend(indices)
            balanced_indices.extend(extra.tolist())
        elif n > target_count:
            # Undersample
            selected = np.random.choice(indices, target_count, replace=False)
            balanced_indices.extend(selected.tolist())
        else:
            balanced_indices.extend(indices)
    
    # Shuffle
    np.random.shuffle(balanced_indices)
    
    balanced_paths = [paths[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    
    logger.info(
        f"Balanced dataset from {len(paths)} to {len(balanced_paths)} samples "
        f"using {method}"
    )
    
    return balanced_paths, balanced_labels


def load_dataset_from_folder(
    root_dir: Union[str, Path],
    real_folder: str = "real",
    fake_folder: str = "fake",
    extensions: Optional[List[str]] = None
) -> Tuple[List[str], List[int]]:
    """
    Load dataset from folder structure with real/fake subdirectories.
    
    Expected structure:
        root_dir/
            real/
                file1.mp4
                file2.mp4
            fake/
                file1.mp4
                file2.mp4
    
    Args:
        root_dir: Root directory
        real_folder: Name of real samples folder
        fake_folder: Name of fake samples folder
        extensions: List of valid file extensions
        
    Returns:
        Tuple of (paths, labels)
    """
    root_dir = Path(root_dir)
    
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png']
    
    extensions = [ext.lower() for ext in extensions]
    
    paths = []
    labels = []
    
    # Load real samples (try multiple folder name variations)
    real_folder_options = [real_folder, real_folder.lower(), real_folder.capitalize(), "Real"]
    real_dir = None
    for folder_name in real_folder_options:
        candidate = root_dir / folder_name
        if candidate.exists():
            real_dir = candidate
            break
    
    if real_dir and real_dir.exists():
        for file in real_dir.iterdir():
            if file.suffix.lower() in extensions:
                paths.append(str(file))
                labels.append(0)
    
    # Load fake samples (try multiple folder name variations)
    fake_folder_options = [fake_folder, fake_folder.lower(), fake_folder.capitalize(), "Fake"]
    fake_dir = None
    for folder_name in fake_folder_options:
        candidate = root_dir / folder_name
        if candidate.exists():
            fake_dir = candidate
            break
    
    if fake_dir and fake_dir.exists():
        for file in fake_dir.iterdir():
            if file.suffix.lower() in extensions:
                paths.append(str(file))
                labels.append(1)
    
    logger.info(f"Loaded {len(paths)} samples from {root_dir}")
    
    return paths, labels


def load_flat_dataset_from_folder(
    root_dir: Union[str, Path],
    real_prefix: str = "R_",
    fake_prefix: str = "F_",
    extensions: Optional[List[str]] = None
) -> Tuple[List[str], List[int]]:
    """
    Load dataset from a flat folder structure where labels are determined by filename prefix.
    
    Expected structure:
        root_dir/
            F_image1.png  (Fake - starts with F_)
            R_image2.png  (Real - starts with R_)
            
    Args:
        root_dir: Root directory containing images
        real_prefix: Filename prefix for real samples (default "R_")
        fake_prefix: Filename prefix for fake samples (default "F_")
        extensions: List of valid file extensions
        
    Returns:
        Tuple of (paths, labels)
    """
    root_dir = Path(root_dir)
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    extensions = [ext.lower() for ext in extensions]
    
    paths = []
    labels = []
    
    for file in root_dir.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            filename = file.name
            if filename.startswith(fake_prefix):
                paths.append(str(file))
                labels.append(1)  # Fake
            elif filename.startswith(real_prefix):
                paths.append(str(file))
                labels.append(0)  # Real
            # Skip files that don't match either prefix
    
    logger.info(f"Loaded {len(paths)} samples from flat folder {root_dir}")
    
    return paths, labels


def load_combined_datasets(
    dataset_configs: List[Dict],
    extensions: Optional[List[str]] = None
) -> Tuple[List[str], List[int]]:
    """
    Load and combine multiple datasets for training.
    
    Each config dict can have:
        - 'path': Root path
        - 'type': 'folder' (Real/Fake subfolders) or 'flat' (prefix-based) or 'fake_only' or 'real_only'
        - 'label': For 'fake_only'/'real_only', specify the label (0=real, 1=fake)
        
    Args:
        dataset_configs: List of dataset configuration dicts
        extensions: List of valid file extensions
        
    Returns:
        Tuple of (combined_paths, combined_labels)
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    extensions = [ext.lower() for ext in extensions]
    
    all_paths = []
    all_labels = []
    
    for config in dataset_configs:
        root_dir = Path(config['path'])
        dataset_type = config.get('type', 'folder')
        
        if dataset_type == 'folder':
            paths, labels = load_dataset_from_folder(root_dir, extensions=extensions)
        elif dataset_type == 'flat':
            paths, labels = load_flat_dataset_from_folder(
                root_dir, 
                real_prefix=config.get('real_prefix', 'R_'),
                fake_prefix=config.get('fake_prefix', 'F_'),
                extensions=extensions
            )
        elif dataset_type in ('fake_only', 'real_only'):
            label = 1 if dataset_type == 'fake_only' else 0
            paths = []
            labels = []
            for file in root_dir.iterdir():
                if file.is_file() and file.suffix.lower() in extensions:
                    paths.append(str(file))
                    labels.append(label)
        else:
            logger.warning(f"Unknown dataset type: {dataset_type}")
            continue
        
        all_paths.extend(paths)
        all_labels.extend(labels)
        logger.info(f"Added {len(paths)} samples from {root_dir} (type={dataset_type})")
    
    logger.info(f"Total combined: {len(all_paths)} samples")
    
    return all_paths, all_labels


def load_faceforensics_dataset(
    root_dir: Union[str, Path],
    compression: str = "c23",
    manipulation_methods: Optional[List[str]] = None
) -> Tuple[List[str], List[int]]:
    """
    Load FaceForensics++ dataset.
    
    Args:
        root_dir: Root directory of FaceForensics++ dataset
        compression: Compression level (c23, c40, raw)
        manipulation_methods: List of manipulation methods to include
            (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
            
    Returns:
        Tuple of (paths, labels)
    """
    root_dir = Path(root_dir)
    
    if manipulation_methods is None:
        manipulation_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    
    paths = []
    labels = []
    
    # Load original (real) videos
    original_dir = root_dir / "original_sequences" / "youtube" / compression / "videos"
    if original_dir.exists():
        for video in original_dir.glob("*.mp4"):
            paths.append(str(video))
            labels.append(0)
    
    # Load manipulated (fake) videos
    for method in manipulation_methods:
        method_dir = root_dir / "manipulated_sequences" / method / compression / "videos"
        if method_dir.exists():
            for video in method_dir.glob("*.mp4"):
                paths.append(str(video))
                labels.append(1)
    
    logger.info(
        f"Loaded FaceForensics++ dataset: {len(paths)} samples "
        f"(compression={compression}, methods={manipulation_methods})"
    )
    
    return paths, labels


def save_split_info(
    splits: Dict[str, Tuple[List[str], List[int]]],
    output_path: Union[str, Path]
) -> None:
    """
    Save split information to JSON file.
    
    Args:
        splits: Dictionary with train/val/test splits
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    split_data = {}
    for split_name, (paths, labels) in splits.items():
        split_data[split_name] = {
            'paths': paths,
            'labels': labels,
            'count': len(paths),
            'label_distribution': dict(Counter(labels))
        }
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    logger.info(f"Saved split info to {output_path}")


def load_split_info(
    input_path: Union[str, Path]
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Load split information from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary with train/val/test splits
    """
    with open(input_path, 'r') as f:
        split_data = json.load(f)
    
    splits = {}
    for split_name, data in split_data.items():
        splits[split_name] = (data['paths'], data['labels'])
    
    return splits

