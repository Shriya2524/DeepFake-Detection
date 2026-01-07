"""
Training script for the spatiotemporal deepfake detection model.

Usage:
    python -m src.training.train_spatiotemporal --config config/default_config.yaml
"""

import argparse
from pathlib import Path

import torch

from ..datasets.video_dataset import VideoDataset, create_video_dataloaders
from ..datasets.data_utils import load_dataset_from_folder, create_data_splits
from ..models.spatiotemporal_model import create_spatiotemporal_model
from ..evaluation.metrics import compute_classification_metrics
from ..utils.config import load_config, init_config
from ..utils.device import seed_everything
from ..utils.logger import setup_logger, get_logger
from .trainer import Trainer, TrainingConfig
from .losses import FocalLoss


def main():
    parser = argparse.ArgumentParser(description="Train spatiotemporal model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
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
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    init_config(args.config)
    
    # Setup logging
    logger = setup_logger(
        name="train_spatiotemporal",
        level=config.logging.level,
        log_dir=config.logging.log_dir
    )
    
    logger.info("=" * 60)
    logger.info("Spatiotemporal Model Training")
    logger.info("=" * 60)
    
    # Set random seed
    seed_everything(config.seed)
    
    # Prepare data paths
    data_dir = args.data_dir or config.dataset.root_path
    logger.info(f"Loading data from: {data_dir}")
    
    # Load dataset
    video_paths, labels = load_dataset_from_folder(
        root_dir=data_dir,
        real_folder="real",
        fake_folder="fake",
        extensions=config.dataset.video_extensions
    )
    
    if not video_paths:
        logger.error("No videos found in data directory!")
        logger.info("Expected structure: data_dir/real/*.mp4 and data_dir/fake/*.mp4")
        return
    
    logger.info(f"Found {len(video_paths)} videos")
    
    # Limit samples if configured
    max_samples = config.dataset.max_samples
    if max_samples and max_samples < len(video_paths):
        logger.info(f"Limiting to {max_samples} samples")
    
    # Create splits
    splits = create_data_splits(
        video_paths,
        labels,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
        max_samples=max_samples,
        seed=config.seed
    )
    
    train_paths, train_labels = splits['train']
    val_paths, val_labels = splits['val']
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create dataloaders
    batch_size = args.batch_size or config.training.batch_size
    
    dataloaders = create_video_dataloaders(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        batch_size=batch_size,
        val_batch_size=config.training.val_batch_size,
        num_workers=config.training.num_workers,
        target_fps=config.preprocessing.target_fps,
        max_frames=config.preprocessing.max_frames,
        image_size=config.preprocessing.image_size,
        extract_faces=True,
        compute_flow=config.spatiotemporal_model.use_optical_flow
    )
    
    # Create model
    model = create_spatiotemporal_model(config)
    logger.info(f"Model created: {model.__class__.__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup loss and metrics
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    def metrics_fn(preds, targets):
        return compute_classification_metrics(preds, targets)
    
    # Training configuration
    training_config = TrainingConfig(
        epochs=args.epochs or config.training.epochs,
        learning_rate=args.lr or config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer=config.training.optimizer,
        scheduler=config.training.scheduler,
        gradient_clip=config.training.gradient_clip,
        mixed_precision=config.training.mixed_precision,
        early_stopping_patience=config.training.early_stopping_patience,
        save_best_only=config.training.save_best_only,
        checkpoint_dir=str(Path(config.paths.checkpoints) / "spatiotemporal")
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        device=args.device,
        criterion=criterion,
        metrics_fn=metrics_fn
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.fit(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val']
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
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

