"""
Generic trainer class for deepfake detection models.

Provides a flexible training loop with:
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- Metric logging
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.device import DeviceManager


logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 10
    save_best_only: bool = True
    save_frequency: int = 5
    checkpoint_dir: str = "checkpoints/"
    log_interval: int = 10


class Trainer:
    """
    Generic trainer for PyTorch models.
    
    Example:
        trainer = Trainer(model, config, device="cuda")
        trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Union[TrainingConfig, dict],
        device: str = "auto",
        criterion: Optional[nn.Module] = None,
        metrics_fn: Optional[Callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to train on
            criterion: Loss function (default: CrossEntropyLoss)
            metrics_fn: Function to compute metrics from predictions and targets
        """
        # Handle config
        if isinstance(config, dict):
            self.config = TrainingConfig(**config)
        else:
            self.config = config
        
        # Device setup
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        
        # Model
        self.model = self.device_manager.prepare_model(model)
        
        # Loss function
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Metrics function
        self.metrics_fn = metrics_fn
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.mixed_precision and self.device.type == "cuda" else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rate': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == "adam":
            return Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler.lower() == "step":
            return StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config)
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            if val_metrics:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                logger.info(f"Val Metrics: {metrics_str}")
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if is_best or not self.config.save_best_only:
                if is_best or (epoch + 1) % self.config.save_frequency == 0:
                    self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info("Training completed")
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> tuple:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = self.device_manager.to_device(batch)
            
            # Get inputs and targets
            inputs, targets = self._unpack_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self._forward(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward(inputs)
                loss = self._compute_loss(outputs, targets)
                
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            if isinstance(outputs, dict):
                preds = outputs.get('logits', outputs.get('probs'))
            else:
                preds = outputs
            
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = {}
        if self.metrics_fn:
            metrics = self.metrics_fn(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> tuple:
        """Run one validation epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
            
            for batch in pbar:
                batch = self.device_manager.to_device(batch)
                inputs, targets = self._unpack_batch(batch)
                
                if self.scaler:
                    with autocast():
                        outputs = self._forward(inputs)
                        loss = self._compute_loss(outputs, targets)
                else:
                    outputs = self._forward(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                
                if isinstance(outputs, dict):
                    preds = outputs.get('logits', outputs.get('probs'))
                else:
                    preds = outputs
                
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = {}
        if self.metrics_fn:
            metrics = self.metrics_fn(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _unpack_batch(self, batch: Union[dict, tuple]) -> tuple:
        """Unpack batch into inputs and targets."""
        if isinstance(batch, dict):
            targets = batch.get('label', batch.get('labels'))
            
            # Check for different input types
            if 'frames' in batch:
                # Video batch
                inputs = {
                    'frames': batch['frames'],
                    'flow': batch.get('flow'),
                    'mask': batch.get('mask')
                }
            elif 'multichannel' in batch:
                # Multi-channel image batch
                inputs = batch['multichannel']
            elif 'image' in batch:
                # Standard image batch
                inputs = batch['image']
            else:
                raise ValueError("Unknown batch format")
        else:
            inputs, targets = batch
        
        return inputs, targets
    
    def _forward(self, inputs: Union[torch.Tensor, dict]) -> Union[torch.Tensor, dict]:
        """Forward pass through model."""
        if isinstance(inputs, dict):
            return self.model(**inputs)
        else:
            return self.model(inputs)
    
    def _compute_loss(
        self,
        outputs: Union[torch.Tensor, dict],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss from outputs."""
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('probs'))
        else:
            logits = outputs
        
        return self.criterion(logits, targets)
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None) -> None:
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch+1}.pt"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']+1})")
    
    def get_model(self) -> nn.Module:
        """Get the trained model."""
        # Unwrap DataParallel if needed
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

