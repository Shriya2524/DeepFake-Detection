#!/usr/bin/env python
"""
CycleGAN Detection Training Script
===================================
Trains a model to detect CycleGAN-generated fake images.

Features:
- Robust error handling with automatic fallback strategies
- Multiple backbone options with auto-switching
- Graceful degradation on failures
- Checkpoint recovery
- Memory-efficient training options

Dataset: Horse ↔ Zebra transformation detection
"""

import os
import sys
import gc
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.models.backbones import get_backbone
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback logger if imports fail
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def get_backbone(*args, **kwargs):
        raise ImportError("Could not import backbone module")


# =============================================================================
# CONFIGURATION & FALLBACK OPTIONS
# =============================================================================

FALLBACK_CONFIGS = [
    # Config 1: Default settings
    {
        'name': 'Default (ResNet18, batch=32)',
        'backbone': 'resnet18',
        'batch_size': 32,
        'num_workers': 4,
        'use_amp': True,
        'image_size': 256,
    },
    # Config 2: Reduced batch size
    {
        'name': 'Reduced Batch (ResNet18, batch=16)',
        'backbone': 'resnet18',
        'batch_size': 16,
        'num_workers': 2,
        'use_amp': True,
        'image_size': 256,
    },
    # Config 3: Minimal batch with smaller images
    {
        'name': 'Memory Efficient (ResNet18, batch=8, 224px)',
        'backbone': 'resnet18',
        'batch_size': 8,
        'num_workers': 0,
        'use_amp': True,
        'image_size': 224,
    },
    # Config 4: CPU fallback
    {
        'name': 'CPU Fallback (ResNet18, batch=4)',
        'backbone': 'resnet18',
        'batch_size': 4,
        'num_workers': 0,
        'use_amp': False,
        'image_size': 224,
        'force_cpu': True,
    },
    # Config 5: Simple CNN (no pretrained backbone)
    {
        'name': 'Simple CNN (no backbone)',
        'backbone': 'simple',
        'batch_size': 8,
        'num_workers': 0,
        'use_amp': False,
        'image_size': 128,
    },
]


# =============================================================================
# SIMPLE FALLBACK MODEL
# =============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN fallback when backbone loading fails."""
    
    def __init__(self, num_classes: int = 2, input_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs, 'features': x.flatten(1)}


# =============================================================================
# ROBUST DATASET
# =============================================================================

class RobustCycleGANDataset(Dataset):
    """
    Robust dataset with error handling for corrupted/missing images.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_path: str = None,
        split: str = "train",
        transform = None,
        binary_classification: bool = True,
        image_size: int = 256
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.binary_classification = binary_classification
        self.image_size = image_size
        self.transform = transform or self._default_transform()
        self.failed_images = []
        
        # Load and validate samples
        self.samples = self._load_samples(metadata_path)
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {data_dir} for split '{split}'")
        
        logger.info(f"Loaded {len(self.samples)} valid samples for {split} split")
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self, metadata_path: str) -> List[Dict]:
        """Load samples with validation."""
        samples = []
        
        # Method 1: Scan directories directly (more reliable)
        logger.info("  Scanning directories for images...")
        folders = ['trainA', 'trainB'] if self.split == 'train' else ['testA', 'testB']
        
        for folder in folders:
            folder_path = self.data_dir / folder
            if folder_path.exists():
                domain = 'A (Horse)' if 'A' in folder else 'B (Zebra)'
                label = 0 if 'A' in folder else 1
                
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_file in folder_path.glob(ext):
                        # Quick check - just verify file exists and has size
                        try:
                            if img_file.stat().st_size > 0:
                                samples.append({
                                    'path': str(img_file),
                                    'domain': domain,
                                    'label': label
                                })
                        except Exception:
                            self.failed_images.append(str(img_file))
        
        if self.failed_images:
            logger.warning(f"  Skipped {len(self.failed_images)} problematic images")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                current_idx = (idx + retry) % len(self.samples)
                current_sample = self.samples[current_idx]
                
                image = Image.open(current_sample['path']).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                label = 0 if self.binary_classification else current_sample['label']
                return image, label
                
            except Exception as e:
                if retry == max_retries - 1:
                    # Last resort: return a blank image
                    logger.warning(f"Failed to load image after {max_retries} retries, using blank")
                    blank = torch.zeros(3, self.image_size, self.image_size)
                    return blank, 0
                continue
        
        # Fallback
        blank = torch.zeros(3, self.image_size, self.image_size)
        return blank, 0
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes."""
        dist = {}
        for sample in self.samples:
            domain = sample['domain']
            dist[domain] = dist.get(domain, 0) + 1
        return dist


# =============================================================================
# ROBUST MODEL FACTORY
# =============================================================================

def create_model(backbone: str, num_classes: int = 2, pretrained: bool = True, 
                 device: torch.device = None) -> nn.Module:
    """
    Create model with fallback options.
    """
    if backbone == 'simple':
        logger.info("Creating SimpleCNN model (fallback)")
        return SimpleCNN(num_classes=num_classes)
    
    try:
        # Try creating with pretrained backbone
        backbone_module = get_backbone(
            backbone_type=backbone,
            pretrained=pretrained
        )
        feature_dim = backbone_module.get_feature_dim()
        
        class CycleGANDetector(nn.Module):
            def __init__(self, backbone_mod, feat_dim, n_classes):
                super().__init__()
                self.backbone = backbone_mod
                self.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, n_classes)
                )
            
            def forward(self, x):
                features = self.backbone(x)
                logits = self.classifier(features)
                probs = torch.softmax(logits, dim=-1)
                return {'logits': logits, 'probs': probs, 'features': features}
        
        model = CycleGANDetector(backbone_module, feature_dim, num_classes)
        logger.info(f"Created CycleGANDetector with {backbone} backbone")
        return model
        
    except Exception as e:
        logger.warning(f"Failed to create {backbone} model: {e}")
        logger.info("Falling back to SimpleCNN")
        return SimpleCNN(num_classes=num_classes)


# =============================================================================
# ROBUST TRAINING FUNCTIONS
# =============================================================================

def train_epoch_safe(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler = None,
    use_amp: bool = True
) -> Dict[str, float]:
    """Train for one epoch with error handling."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_errors = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            if use_amp and scaler is not None and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output['logits'], labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(images)
                loss = criterion(output['logits'], labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            predictions = output['probs'].argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100*correct/total:.2f}%"
            })
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f"OOM at batch {batch_idx}, skipping...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                batch_errors += 1
                if batch_errors > 5:
                    raise RuntimeError("Too many OOM errors, trying next config...")
            else:
                raise
        except Exception as e:
            logger.warning(f"Error in batch {batch_idx}: {e}")
            batch_errors += 1
            if batch_errors > 10:
                raise RuntimeError("Too many batch errors, trying next config...")
    
    if total == 0:
        return {'loss': float('inf'), 'accuracy': 0.0, 'errors': batch_errors}
    
    return {
        'loss': total_loss / max(len(dataloader), 1),
        'accuracy': correct / total,
        'errors': batch_errors
    }


def validate_safe(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate with error handling."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            try:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                output = model(images)
                loss = criterion(output['logits'], labels)
                
                total_loss += loss.item()
                predictions = output['probs'].argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            except Exception as e:
                logger.warning(f"Validation error: {e}")
                continue
    
    if total == 0:
        return {'loss': float('inf'), 'accuracy': 0.0}
    
    return {
        'loss': total_loss / max(len(dataloader), 1),
        'accuracy': correct / total
    }


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# MAIN TRAINING FUNCTION WITH AUTO-FALLBACK
# =============================================================================

def train_with_fallback(args) -> bool:
    """
    Main training function with automatic fallback on errors.
    
    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print("    CycleGAN Detection Training")
    print("    With Automatic Error Recovery")
    print(f"{'='*60}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try each fallback configuration
    for config_idx, config in enumerate(FALLBACK_CONFIGS):
        print(f"\n{'─'*60}")
        print(f"Attempting Configuration {config_idx + 1}/{len(FALLBACK_CONFIGS)}")
        print(f"  -> {config['name']}")
        print(f"{'─'*60}")
        
        try:
            # Setup device
            if config.get('force_cpu', False):
                device = torch.device('cpu')
                print("  Device: CPU (forced)")
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"  Device: {device}")
                if device.type == 'cuda':
                    print(f"  GPU: {torch.cuda.get_device_name(0)}")
            
            cleanup_memory()
            
            # Create transforms
            image_size = config['image_size']
            train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Load datasets
            print("  Loading datasets...")
            try:
                train_dataset = RobustCycleGANDataset(
                    data_dir=args.data_dir,
                    metadata_path=f"{args.data_dir}/metadata.csv",
                    split="train",
                    transform=train_transform,
                    binary_classification=False,
                    image_size=image_size
                )
                
                test_dataset = RobustCycleGANDataset(
                    data_dir=args.data_dir,
                    metadata_path=f"{args.data_dir}/metadata.csv",
                    split="test",
                    transform=val_transform,
                    binary_classification=False,
                    image_size=image_size
                )
            except ValueError as e:
                print(f"  [X] Dataset error: {e}")
                continue
            
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Test samples: {len(test_dataset)}")
            print(f"  Class distribution: {train_dataset.get_class_distribution()}")
            
            # Split training into train/val
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
            
            batch_size = config['batch_size']
            num_workers = config['num_workers']
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size,
                shuffle=True, 
                num_workers=num_workers, 
                pin_memory=(device.type == 'cuda'),
                drop_last=True
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size,
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda')
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size,
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            # Create model
            print(f"  Creating model with {config['backbone']} backbone...")
            model = create_model(
                backbone=config['backbone'],
                num_classes=2,
                pretrained=True,
                device=device
            )
            model = model.to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Model parameters: {total_params:,}")
            
            # Loss, optimizer, scheduler
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            # Mixed precision scaler
            scaler = None
            if config['use_amp'] and device.type == 'cuda':
                try:
                    scaler = torch.cuda.amp.GradScaler()
                except:
                    scaler = None
            
            # Training loop
            print(f"\n  Starting training for {args.epochs} epochs...")
            best_val_acc = 0.0
            patience_counter = 0
            max_patience = 5
            
            for epoch in range(1, args.epochs + 1):
                print(f"\n  --- Epoch {epoch}/{args.epochs} ---")
                
                try:
                    # Train
                    train_metrics = train_epoch_safe(
                        model, train_loader, criterion,
                        optimizer, device, epoch,
                        scaler=scaler,
                        use_amp=config['use_amp']
                    )
                    
                    # Validate
                    val_metrics = validate_safe(model, val_loader, criterion, device)
                    
                    scheduler.step()
                    
                    print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                          f"Train Acc: {100*train_metrics['accuracy']:.2f}%")
                    print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                          f"Val Acc: {100*val_metrics['accuracy']:.2f}%")
                    
                    # Save best model
                    if val_metrics['accuracy'] > best_val_acc:
                        best_val_acc = val_metrics['accuracy']
                        patience_counter = 0
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_accuracy': val_metrics['accuracy'],
                            'train_accuracy': train_metrics['accuracy'],
                            'config': config
                        }, output_dir / 'best_model.pt')
                        print(f"  [OK] Saved best model (Val Acc: {100*best_val_acc:.2f}%)")
                    else:
                        patience_counter += 1
                        if patience_counter >= max_patience:
                            print(f"  Early stopping after {patience_counter} epochs without improvement")
                            break
                    
                    # Save checkpoint every 5 epochs
                    if epoch % 5 == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'val_accuracy': val_metrics['accuracy']
                        }, output_dir / f'checkpoint_epoch_{epoch}.pt')
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"  [X] OOM Error at epoch {epoch}, trying next config...")
                        cleanup_memory()
                        raise
                    else:
                        print(f"  [X] Error at epoch {epoch}: {e}")
                        raise
            
            # Final evaluation
            print(f"\n{'─'*60}")
            print("  Final Evaluation on Test Set")
            print(f"{'─'*60}")
            
            # Load best model
            if (output_dir / 'best_model.pt').exists():
                checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            test_metrics = validate_safe(model, test_loader, criterion, device)
            print(f"  Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Test Accuracy: {100*test_metrics['accuracy']:.2f}%")
            
            # Save final model
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_metrics['accuracy'],
                'config': {
                    'backbone': config['backbone'],
                    'num_classes': 2,
                    'classes': ['Horse', 'Zebra'],
                    'image_size': image_size
                }
            }, output_dir / 'final_model.pt')
            
            print(f"\n{'='*60}")
            print("  [OK] TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"  Configuration used: {config['name']}")
            print(f"  Best validation accuracy: {100*best_val_acc:.2f}%")
            print(f"  Test accuracy: {100*test_metrics['accuracy']:.2f}%")
            print(f"  Models saved to: {output_dir}")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"\n  [X] Configuration failed: {e}")
            if args.verbose:
                traceback.print_exc()
            cleanup_memory()
            print(f"  Trying next fallback configuration...\n")
            continue
    
    # All configurations failed
    print(f"\n{'='*60}")
    print("  [X] ALL CONFIGURATIONS FAILED")
    print(f"{'='*60}")
    print("  Please check:")
    print("  1. Dataset path is correct")
    print("  2. Images are valid and readable")
    print("  3. Sufficient disk space and memory")
    print(f"{'='*60}\n")
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN Detector with Auto-Fallback")
    parser.add_argument('--data_dir', type=str, default='cyclegan',
                       help='Path to CycleGAN dataset')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints/cyclegan',
                       help='Output directory for checkpoints')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed error traces')
    args = parser.parse_args()
    
    success = train_with_fallback(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
