"""
Training script for the GAN fingerprint detection model.

Usage:
    python -m src.training.train_gan_fingerprint --config config/default_config.yaml
"""

import argparse
from pathlib import Path

import torch

from ..datasets.image_dataset import ImageDataset, create_image_dataloaders
from ..datasets.data_utils import load_dataset_from_folder, create_data_splits, load_combined_datasets
from ..models.gan_fingerprint_model import create_gan_fingerprint_model
from ..evaluation.metrics import compute_classification_metrics
from ..utils.config import load_config, init_config
from ..utils.device import seed_everything
from ..utils.logger import setup_logger, get_logger
from .trainer import Trainer, TrainingConfig
from .losses import FocalLoss


def main():
    parser = argparse.ArgumentParser(description="Train GAN fingerprint model")
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
        "--use-videos",
        action="store_true",
        help="Extract frames from videos instead of using images"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Use combined dataset mode (multiple sources)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    init_config(args.config)
    
    # Setup logging
    logger = setup_logger(
        name="train_gan_fingerprint",
        level=config.logging.level,
        log_dir=config.logging.log_dir
    )
    
    logger.info("=" * 60)
    logger.info("GAN Fingerprint Model Training")
    logger.info("=" * 60)
    
    # Set random seed
    seed_everything(config.seed)
    
    # Prepare data paths
    data_dir = args.data_dir or config.dataset.root_path
    logger.info(f"Loading data from: {data_dir}")
    
    # Load dataset
    if args.use_videos:
        extensions = config.dataset.video_extensions
    else:
        extensions = config.dataset.image_extensions
    
    # Check if using combined dataset mode
    if args.combined and hasattr(config, 'data') and hasattr(config.data, 'datasets'):
        # Multi-source combined dataset mode
        logger.info("Using combined dataset mode")
        dataset_configs = []
        for ds in config.data.datasets:
            dataset_configs.append({
                'path': ds.path,
                'type': ds.type,
            })
        image_paths, labels = load_combined_datasets(dataset_configs, extensions=extensions)
    else:
        # Standard single folder mode
        image_paths, labels = load_dataset_from_folder(
            root_dir=data_dir,
            real_folder="real",
            fake_folder="fake",
            extensions=extensions
        )
    
    if not image_paths:
        logger.error("No images/videos found in data directory!")
        logger.info("Expected structure: data_dir/real/*.jpg and data_dir/fake/*.jpg")
        return
    
    logger.info(f"Found {len(image_paths)} samples")
    
    # Limit samples if configured
    max_samples = config.dataset.max_samples
    if max_samples and max_samples < len(image_paths):
        logger.info(f"Limiting to {max_samples} samples")
    
    # Create splits
    splits = create_data_splits(
        image_paths,
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
    
    dataloaders = create_image_dataloaders(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        batch_size=batch_size,
        val_batch_size=config.training.val_batch_size,
        num_workers=config.training.num_workers,
        image_size=config.preprocessing.image_size,
        extract_faces=True,
        use_multichannel=True,
        use_ycbcr=config.gan_fingerprint_model.use_ycbcr,
        use_frequency=config.gan_fingerprint_model.use_frequency,
        use_edges=config.gan_fingerprint_model.use_edge_features
    )
    
    # Create model
    model = create_gan_fingerprint_model(config)
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
        checkpoint_dir=str(Path(config.paths.checkpoints) / "gan_fingerprint")
    )
    
    # Create trainer - use multichannel input
    class MultiChannelTrainer(Trainer):
        def _unpack_batch(self, batch):
            targets = batch['label']
            inputs = batch['multichannel']  # Use multi-channel input
            return inputs, targets
    
    trainer = MultiChannelTrainer(
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

