"""
Quick training script for PGGAN dataset.
Combines PGGAN fake images with StyleGAN real images.

Usage:
    python train_pggan.py
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.image_dataset import ImageDataset, create_image_dataloaders
from src.datasets.data_utils import create_data_splits, balance_dataset
from src.models.gan_fingerprint_model import create_gan_fingerprint_model
from src.evaluation.metrics import compute_classification_metrics
from src.utils.config import load_config, init_config
from src.utils.device import seed_everything
from src.utils.logger import setup_logger
from src.training.trainer import Trainer, TrainingConfig
from src.training.losses import FocalLoss


def load_images_from_folder(folder_path, label, extensions=None):
    """Load all images from a folder with given label."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    extensions = [ext.lower() for ext in extensions]
    folder = Path(folder_path)
    
    paths = []
    labels = []
    
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            paths.append(str(file))
            labels.append(label)
    
    return paths, labels


def main():
    # Setup
    seed_everything(42)
    
    logger = setup_logger(
        name="train_pggan",
        level="INFO",
        log_dir="outputs/logs"
    )
    
    logger.info("=" * 60)
    logger.info("PGGAN + StyleGAN Combined Training")
    logger.info("=" * 60)
    
    # Load config
    config = load_config("config/experiment_configs/pggan_train.yaml")
    init_config("config/experiment_configs/pggan_train.yaml")
    
    # Collect all images
    all_paths = []
    all_labels = []
    
    # 1. Load PGGAN fake images (label=1)
    pggan_folders = [
        "archive/pggan_v1/pggan_v1/train",
        "archive/pggan_v2/pggan_v2/train",
    ]
    
    for folder in pggan_folders:
        if Path(folder).exists():
            paths, labels = load_images_from_folder(folder, label=1)
            all_paths.extend(paths)
            all_labels.extend(labels)
            logger.info(f"Loaded {len(paths)} FAKE images from {folder}")
    
    # 2. Load StyleGAN real images (label=0)
    real_folder = "stylegan/Final Dataset/Real"
    if Path(real_folder).exists():
        paths, labels = load_images_from_folder(real_folder, label=0)
        all_paths.extend(paths)
        all_labels.extend(labels)
        logger.info(f"Loaded {len(paths)} REAL images from {real_folder}")
    
    # Stats
    label_counts = Counter(all_labels)
    logger.info(f"Total samples: {len(all_paths)}")
    logger.info(f"  Real (0): {label_counts[0]}")
    logger.info(f"  Fake (1): {label_counts[1]}")
    
    # Balance dataset (undersample to match smaller class)
    logger.info("Balancing dataset...")
    all_paths, all_labels = balance_dataset(all_paths, all_labels, method="undersample")
    
    label_counts = Counter(all_labels)
    logger.info(f"After balancing: {len(all_paths)} samples")
    logger.info(f"  Real (0): {label_counts[0]}")
    logger.info(f"  Fake (1): {label_counts[1]}")
    
    # Limit samples for faster training
    max_samples = 4000
    if len(all_paths) > max_samples:
        indices = np.random.choice(len(all_paths), max_samples, replace=False)
        all_paths = [all_paths[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        logger.info(f"Limited to {max_samples} samples")
    
    # Create splits
    splits = create_data_splits(
        all_paths,
        all_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    train_paths, train_labels = splits['train']
    val_paths, val_labels = splits['val']
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create dataloaders
    dataloaders = create_image_dataloaders(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        batch_size=16,
        val_batch_size=32,
        num_workers=0,  # Windows compatibility
        image_size=224,
        extract_faces=True,
        use_multichannel=True,
        use_ycbcr=True,
        use_frequency=True,
        use_edges=True
    )
    
    # Create model
    model = create_gan_fingerprint_model(config)
    logger.info(f"Model: {model.__class__.__name__}")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Loss and metrics
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    def metrics_fn(preds, targets):
        return compute_classification_metrics(preds, targets)
    
    # Training config
    checkpoint_dir = "checkpoints/gan_fingerprint_pggan"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    training_config = TrainingConfig(
        epochs=10,
        learning_rate=0.0001,
        weight_decay=0.0001,
        optimizer="adam",
        scheduler="cosine",
        gradient_clip=1.0,
        mixed_precision=False,
        early_stopping_patience=5,
        save_best_only=False,
        checkpoint_dir=checkpoint_dir
    )
    
    # Custom trainer for multichannel input
    class MultiChannelTrainer(Trainer):
        def _unpack_batch(self, batch):
            targets = batch['label']
            inputs = batch['multichannel']
            return inputs, targets
    
    trainer = MultiChannelTrainer(
        model=model,
        config=training_config,
        device="auto",
        criterion=criterion,
        metrics_fn=metrics_fn
    )
    
    # Train!
    logger.info("Starting training...")
    history = trainer.fit(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val']
    )
    
    # Save final model
    final_model = trainer.get_model()
    final_path = Path(checkpoint_dir) / "final_model.pt"
    torch.save(final_model.state_dict(), final_path)
    logger.info(f"Saved final model: {final_path}")
    
    # Final metrics
    if history['val_metrics']:
        final_metrics = history['val_metrics'][-1]
        logger.info("=" * 60)
        logger.info("Final Validation Metrics:")
        for k, v in final_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    
    logger.info("Training completed!")
    

if __name__ == "__main__":
    main()
