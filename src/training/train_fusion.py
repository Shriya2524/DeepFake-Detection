"""
Training script for the fusion model.

Trains the fusion head using predictions from pre-trained
spatiotemporal and GAN fingerprint models.

Usage:
    python -m src.training.train_fusion --config config/default_config.yaml \
        --spatiotemporal-weights checkpoints/spatiotemporal/best_model.pt \
        --gan-weights checkpoints/gan_fingerprint/best_model.pt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..datasets.video_dataset import VideoDataset, create_video_dataloaders
from ..datasets.data_utils import load_dataset_from_folder, create_data_splits
from ..models.spatiotemporal_model import create_spatiotemporal_model
from ..models.gan_fingerprint_model import create_gan_fingerprint_model
from ..models.fusion_model import create_fusion_model, FusionModel
from ..evaluation.metrics import compute_classification_metrics
from ..utils.config import load_config, init_config
from ..utils.device import seed_everything, get_device
from ..utils.logger import setup_logger, get_logger
from .trainer import Trainer, TrainingConfig


def extract_features(
    video_loader: DataLoader,
    spatiotemporal_model: nn.Module,
    gan_fingerprint_model: nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from both models for fusion training.
    
    Args:
        video_loader: DataLoader with video samples
        spatiotemporal_model: Pre-trained spatiotemporal model
        gan_fingerprint_model: Pre-trained GAN fingerprint model
        device: Device to use
        
    Returns:
        Tuple of (fusion_features, labels)
    """
    spatiotemporal_model.eval()
    gan_fingerprint_model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(video_loader, desc="Extracting features"):
            frames = batch['frames'].to(device)
            flow = batch.get('flow')
            if flow is not None:
                flow = flow.to(device)
            
            labels = batch['label']
            B, T, C, H, W = frames.shape
            
            # Get spatiotemporal prediction
            spatio_output = spatiotemporal_model(frames, flow)
            video_fake_prob = spatio_output['probs'][:, 1]  # (B,)
            
            # Get GAN fingerprint predictions for each frame
            # Note: For simplicity, we use RGB frames directly
            # In practice, you'd use the multi-channel preprocessing
            frames_flat = frames.view(B * T, C, H, W)
            
            # Simple forward through GAN model with RGB
            gan_output = gan_fingerprint_model.forward_rgb_only(frames_flat)
            frame_probs = gan_output['probs'].view(B, T, -1)
            frame_fake_probs = frame_probs[:, :, 1]  # (B, T)
            
            # Compute fusion features
            mean_prob = frame_fake_probs.mean(dim=1)
            std_prob = frame_fake_probs.std(dim=1)
            max_prob = frame_fake_probs.max(dim=1).values
            min_prob = frame_fake_probs.min(dim=1).values
            
            features = torch.stack([
                video_fake_prob,
                mean_prob,
                std_prob,
                max_prob,
                min_prob
            ], dim=1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--spatiotemporal-weights",
        type=str,
        required=True,
        help="Path to pre-trained spatiotemporal model weights"
    )
    parser.add_argument(
        "--gan-weights",
        type=str,
        required=True,
        help="Path to pre-trained GAN fingerprint model weights"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for fusion training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    init_config(args.config)
    
    # Setup logging
    logger = setup_logger(
        name="train_fusion",
        level=config.logging.level,
        log_dir=config.logging.log_dir
    )
    
    logger.info("=" * 60)
    logger.info("Fusion Model Training")
    logger.info("=" * 60)
    
    # Set random seed
    seed_everything(config.seed)
    
    # Setup device
    device = get_device(args.device)
    
    # Load pre-trained models
    logger.info("Loading pre-trained models...")
    
    spatiotemporal_model = create_spatiotemporal_model(config)
    spatiotemporal_model.load_state_dict(
        torch.load(args.spatiotemporal_weights, map_location=device)
    )
    spatiotemporal_model = spatiotemporal_model.to(device)
    spatiotemporal_model.eval()
    logger.info(f"Loaded spatiotemporal model from {args.spatiotemporal_weights}")
    
    gan_fingerprint_model = create_gan_fingerprint_model(config)
    gan_fingerprint_model.load_state_dict(
        torch.load(args.gan_weights, map_location=device)
    )
    gan_fingerprint_model = gan_fingerprint_model.to(device)
    gan_fingerprint_model.eval()
    logger.info(f"Loaded GAN fingerprint model from {args.gan_weights}")
    
    # Load data
    data_dir = args.data_dir or config.dataset.root_path
    logger.info(f"Loading data from: {data_dir}")
    
    video_paths, labels = load_dataset_from_folder(
        root_dir=data_dir,
        real_folder="real",
        fake_folder="fake",
        extensions=config.dataset.video_extensions
    )
    
    if not video_paths:
        logger.error("No videos found in data directory!")
        return
    
    # Create splits
    splits = create_data_splits(
        video_paths,
        labels,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
        max_samples=config.dataset.max_samples,
        seed=config.seed
    )
    
    # Create video dataloaders for feature extraction
    dataloaders = create_video_dataloaders(
        train_paths=splits['train'][0],
        train_labels=splits['train'][1],
        val_paths=splits['val'][0],
        val_labels=splits['val'][1],
        batch_size=4,  # Small batch for feature extraction
        val_batch_size=4,
        num_workers=config.training.num_workers,
        target_fps=config.preprocessing.target_fps,
        max_frames=config.preprocessing.max_frames,
        image_size=config.preprocessing.image_size,
        extract_faces=True,
        compute_flow=config.spatiotemporal_model.use_optical_flow
    )
    
    # Extract features
    logger.info("Extracting features from training data...")
    train_features, train_labels = extract_features(
        dataloaders['train'],
        spatiotemporal_model,
        gan_fingerprint_model,
        device
    )
    
    logger.info("Extracting features from validation data...")
    val_features, val_labels = extract_features(
        dataloaders['val'],
        spatiotemporal_model,
        gan_fingerprint_model,
        device
    )
    
    logger.info(f"Train features shape: {train_features.shape}")
    logger.info(f"Val features shape: {val_features.shape}")
    
    # Create fusion dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    fusion_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    fusion_val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create fusion model
    fusion_model = create_fusion_model(config)
    logger.info(f"Created fusion model")
    
    # Training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=0.001,
        weight_decay=0.01,
        optimizer="adamw",
        scheduler="cosine",
        gradient_clip=1.0,
        mixed_precision=False,  # Fusion model is small, no need for AMP
        early_stopping_patience=10,
        checkpoint_dir=str(Path(config.paths.checkpoints) / "fusion")
    )
    
    # Custom trainer for fusion
    class FusionTrainer(Trainer):
        def _unpack_batch(self, batch):
            features, labels = batch
            return features, labels
        
        def _forward(self, inputs):
            return self.model.forward_features(inputs)
    
    def metrics_fn(preds, targets):
        return compute_classification_metrics(preds, targets)
    
    trainer = FusionTrainer(
        model=fusion_model,
        config=training_config,
        device=args.device,
        criterion=nn.CrossEntropyLoss(),
        metrics_fn=metrics_fn
    )
    
    # Train
    history = trainer.fit(
        train_loader=fusion_train_loader,
        val_loader=fusion_val_loader
    )
    
    # Save final model
    final_model = trainer.get_model()
    final_path = Path(training_config.checkpoint_dir) / "final_model.pt"
    torch.save(final_model.state_dict(), final_path)
    logger.info(f"Saved final model: {final_path}")
    
    # Log final metrics
    if history['val_metrics']:
        final_metrics = history['val_metrics'][-1]
        logger.info("Final Validation Metrics:")
        for k, v in final_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    
    logger.info("Fusion training completed!")


if __name__ == "__main__":
    main()

