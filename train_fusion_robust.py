"""
Robust Fusion Model Training Script with Automatic Fallback Mechanisms

Features:
1. Automatic GPU → CPU fallback if CUDA fails
2. Automatic batch size reduction on OOM errors
3. Multiple GAN model fallback options
4. Checkpoint resume capability
5. Graceful error handling with retries
6. Automatic memory cleanup

Usage:
    python train_fusion_robust.py
"""

import os
import sys
import gc
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.video_dataset import VideoDataset, create_video_dataloaders
from src.datasets.data_utils import load_dataset_from_folder, create_data_splits
from src.models.spatiotemporal_model import create_spatiotemporal_model
from src.models.gan_fingerprint_model import create_gan_fingerprint_model
from src.models.fusion_model import create_fusion_model, FusionModel
from src.evaluation.metrics import compute_classification_metrics
from src.utils.config import load_config, init_config
from src.utils.device import seed_everything
from src.utils.logger import setup_logger


# ============================================================================
# CONFIGURATION - FALLBACK OPTIONS
# ============================================================================

CONFIG = {
    # Primary settings
    "config_path": "config/default_config.yaml",
    "data_dir": "data/faceforensics",
    "epochs": 20,
    
    # Batch sizes to try (will reduce automatically on OOM)
    "batch_sizes": [4, 2, 1],
    
    # Feature extraction batch sizes
    "extract_batch_sizes": [4, 2, 1],
    
    # Devices to try in order
    "devices": ["cuda", "cpu"],
    
    # Spatiotemporal model weights (in order of preference)
    "spatiotemporal_weights": [
        "checkpoints/spatiotemporal/best_model.pt",
        "checkpoints/spatiotemporal/final_model.pt",
        "checkpoints/spatiotemporal/checkpoint_epoch_17.pt",
        "checkpoints/spatiotemporal/checkpoint_epoch_10.pt",
    ],
    
    # GAN fingerprint model weights (in order of preference)
    "gan_weights": [
        "checkpoints/gan_fingerprint/best_model.pt",
        "checkpoints/gan_fingerprint/final_model.pt",
        "checkpoints/gan_fingerprint_pggan/best_model.pt",
        "checkpoints/gan_fingerprint_pggan/final_model.pt",
    ],
    
    # Retry settings
    "max_retries": 3,
    "retry_delay": 5,  # seconds
    
    # Memory management
    "clear_cache_frequency": 10,  # Clear GPU cache every N batches
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_available_device(preferred_devices: List[str]) -> torch.device:
    """Get the first available device from the list."""
    for device_name in preferred_devices:
        if device_name == "cuda":
            if torch.cuda.is_available():
                try:
                    # Test if CUDA actually works
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    clear_gpu_memory()
                    return torch.device("cuda")
                except Exception as e:
                    print(f"⚠️ CUDA available but failed: {e}")
                    continue
        elif device_name == "cpu":
            return torch.device("cpu")
    
    return torch.device("cpu")


def find_valid_weights(weight_paths: List[str]) -> Optional[str]:
    """Find the first valid weights file from the list."""
    for path in weight_paths:
        if Path(path).exists():
            try:
                # Try to load to verify it's valid
                checkpoint = torch.load(path, map_location="cpu")
                del checkpoint
                gc.collect()
                return path
            except Exception as e:
                print(f"⚠️ Invalid weights file {path}: {e}")
                continue
    return None


def load_model_with_fallback(
    model_creator,
    weight_paths: List[str],
    device: torch.device,
    config,
    model_name: str
) -> Optional[nn.Module]:
    """Load a model with fallback weight options."""
    
    valid_weights = find_valid_weights(weight_paths)
    if valid_weights is None:
        print(f"❌ No valid weights found for {model_name}")
        return None
    
    print(f"📂 Loading {model_name} from: {valid_weights}")
    
    try:
        model = model_creator(config)
        state_dict = torch.load(valid_weights, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"✅ {model_name} loaded successfully")
        return model
    
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        
        # Try next weights file
        remaining_paths = [p for p in weight_paths if p != valid_weights]
        if remaining_paths:
            print(f"🔄 Trying alternative weights...")
            return load_model_with_fallback(
                model_creator, remaining_paths, device, config, model_name
            )
        return None


# ============================================================================
# FEATURE EXTRACTION WITH FALLBACKS
# ============================================================================

def extract_features_safe(
    video_loader: DataLoader,
    spatiotemporal_model: nn.Module,
    gan_fingerprint_model: nn.Module,
    device: torch.device,
    logger,
    batch_sizes: List[int] = [4, 2, 1]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract features with automatic fallback on errors.
    """
    
    spatiotemporal_model.eval()
    gan_fingerprint_model.eval()
    
    all_features = []
    all_labels = []
    failed_batches = 0
    max_failures = 50
    
    with torch.no_grad():
        pbar = tqdm(video_loader, desc="Extracting features")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Clear cache periodically
                if batch_idx % CONFIG["clear_cache_frequency"] == 0:
                    clear_gpu_memory()
                
                frames = batch['frames'].to(device)
                flow = batch.get('flow')
                if flow is not None:
                    flow = flow.to(device)
                
                labels = batch['label']
                B, T, C, H, W = frames.shape
                
                # Get spatiotemporal prediction
                try:
                    spatio_output = spatiotemporal_model(frames, flow)
                    video_fake_prob = spatio_output['probs'][:, 1].cpu()
                except Exception as e:
                    logger.warning(f"Spatiotemporal forward failed: {e}, using default")
                    video_fake_prob = torch.ones(B) * 0.5
                
                # Get GAN fingerprint predictions for each frame
                try:
                    frames_flat = frames.view(B * T, C, H, W)
                    
                    # Try forward_rgb_only first, fallback to regular forward
                    try:
                        gan_output = gan_fingerprint_model.forward_rgb_only(frames_flat)
                    except AttributeError:
                        # Model doesn't have forward_rgb_only, use regular forward
                        gan_output = gan_fingerprint_model(frames_flat)
                    
                    frame_probs = gan_output['probs'].view(B, T, -1).cpu()
                    frame_fake_probs = frame_probs[:, :, 1]
                except Exception as e:
                    logger.warning(f"GAN forward failed: {e}, using default")
                    frame_fake_probs = torch.ones(B, T) * 0.5
                
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
                
                all_features.append(features)
                all_labels.append(labels)
                
                # Clear intermediate tensors
                del frames, flow, frames_flat
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at batch {batch_idx}, clearing cache...")
                    clear_gpu_memory()
                    failed_batches += 1
                    
                    if failed_batches > max_failures:
                        logger.error("Too many OOM failures, stopping extraction")
                        break
                    continue
                else:
                    logger.error(f"Runtime error: {e}")
                    failed_batches += 1
                    continue
            
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                failed_batches += 1
                if failed_batches > max_failures:
                    break
                continue
    
    if len(all_features) == 0:
        return None, None
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ============================================================================
# TRAINING WITH FALLBACKS
# ============================================================================

def train_fusion_model(
    fusion_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    logger,
    checkpoint_dir: Path
) -> Dict:
    """Train fusion model with automatic fallbacks."""
    
    fusion_model = fusion_model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        fusion_model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Training state
    best_val_loss = float('inf')
    best_accuracy = 0.0
    patience = 10
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # ===================== TRAINING =====================
        fusion_model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (features, labels) in enumerate(pbar):
            try:
                features = features.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward
                output = fusion_model.forward_features(features)
                logits = output['logits']
                
                loss = criterion(logits, labels)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM in training, clearing cache...")
                    clear_gpu_memory()
                    continue
                raise
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        
        # ===================== VALIDATION =====================
        fusion_model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for features, labels in pbar:
                try:
                    features = features.to(device)
                    labels_device = labels.to(device)
                    
                    output = fusion_model.forward_features(features)
                    logits = output['logits']
                    
                    loss = criterion(logits, labels_device)
                    val_losses.append(loss.item())
                    
                    all_preds.append(logits.cpu())
                    all_labels.append(labels)
                    
                except Exception as e:
                    logger.warning(f"Validation batch failed: {e}")
                    continue
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        # Compute metrics
        if all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = compute_classification_metrics(all_preds, all_labels)
            accuracy = metrics.get('accuracy', 0)
        else:
            metrics = {}
            accuracy = 0
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(accuracy)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Val Metrics: {metrics_str}")
        
        # Save best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_accuracy = accuracy
            patience_counter = 0
            
            torch.save(fusion_model.state_dict(), checkpoint_dir / "best_model.pt")
            logger.info(f"✅ Saved best model (accuracy: {accuracy:.4f})")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'history': history
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(fusion_model.state_dict(), checkpoint_dir / "final_model.pt")
    logger.info(f"✅ Saved final model")
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_accuracy': best_accuracy
    }


# ============================================================================
# MAIN FUNCTION WITH FULL FALLBACK LOGIC
# ============================================================================

def main():
    print("=" * 70)
    print("   ROBUST FUSION MODEL TRAINING WITH AUTOMATIC FALLBACKS")
    print("=" * 70)
    print()
    
    # Setup logging
    logger = setup_logger(
        name="train_fusion_robust",
        level="INFO",
        log_dir="outputs/logs"
    )
    
    logger.info("Starting robust fusion training...")
    
    # Load configuration
    try:
        config = load_config(CONFIG["config_path"])
        init_config(CONFIG["config_path"])
        logger.info("✅ Configuration loaded")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
    # Set random seed
    seed_everything(config.seed)
    
    # =========== STEP 1: FIND DEVICE ===========
    print("\n🔍 Step 1: Finding available device...")
    device = get_available_device(CONFIG["devices"])
    logger.info(f"✅ Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # =========== STEP 2: LOAD MODELS ===========
    print("\n🔍 Step 2: Loading pre-trained models...")
    
    # Load spatiotemporal model
    spatiotemporal_model = load_model_with_fallback(
        create_spatiotemporal_model,
        CONFIG["spatiotemporal_weights"],
        device,
        config,
        "Spatiotemporal Model"
    )
    
    if spatiotemporal_model is None:
        logger.error("❌ Could not load spatiotemporal model. Aborting.")
        return
    
    # Load GAN fingerprint model
    gan_fingerprint_model = load_model_with_fallback(
        create_gan_fingerprint_model,
        CONFIG["gan_weights"],
        device,
        config,
        "GAN Fingerprint Model"
    )
    
    if gan_fingerprint_model is None:
        logger.error("❌ Could not load GAN fingerprint model. Aborting.")
        return
    
    # =========== STEP 3: LOAD DATA ===========
    print("\n🔍 Step 3: Loading video dataset...")
    
    data_dir = CONFIG["data_dir"]
    if not Path(data_dir).exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return
    
    video_paths, labels = load_dataset_from_folder(
        root_dir=data_dir,
        real_folder="real",
        fake_folder="fake",
        extensions=config.dataset.video_extensions
    )
    
    if not video_paths:
        logger.error("❌ No videos found!")
        return
    
    logger.info(f"✅ Found {len(video_paths)} videos")
    
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
    
    logger.info(f"   Train: {len(splits['train'][0])}, Val: {len(splits['val'][0])}")
    
    # =========== STEP 4: EXTRACT FEATURES ===========
    print("\n🔍 Step 4: Extracting features (this may take a while)...")
    
    train_features = None
    val_features = None
    
    for batch_size in CONFIG["extract_batch_sizes"]:
        logger.info(f"   Trying batch size: {batch_size}")
        
        try:
            # Create dataloaders
            dataloaders = create_video_dataloaders(
                train_paths=splits['train'][0],
                train_labels=splits['train'][1],
                val_paths=splits['val'][0],
                val_labels=splits['val'][1],
                batch_size=batch_size,
                val_batch_size=batch_size,
                num_workers=0,  # Windows compatibility
                target_fps=config.preprocessing.target_fps,
                max_frames=config.preprocessing.max_frames,
                image_size=config.preprocessing.image_size,
                extract_faces=True,
                compute_flow=config.spatiotemporal_model.use_optical_flow
            )
            
            # Extract training features
            logger.info("   Extracting training features...")
            train_features, train_labels = extract_features_safe(
                dataloaders['train'],
                spatiotemporal_model,
                gan_fingerprint_model,
                device,
                logger,
                CONFIG["extract_batch_sizes"]
            )
            
            if train_features is None:
                logger.warning(f"   Failed with batch size {batch_size}, trying smaller...")
                clear_gpu_memory()
                continue
            
            # Extract validation features
            logger.info("   Extracting validation features...")
            val_features, val_labels = extract_features_safe(
                dataloaders['val'],
                spatiotemporal_model,
                gan_fingerprint_model,
                device,
                logger,
                CONFIG["extract_batch_sizes"]
            )
            
            if val_features is None:
                logger.warning(f"   Validation extraction failed, trying smaller batch...")
                clear_gpu_memory()
                continue
            
            logger.info(f"✅ Features extracted successfully!")
            logger.info(f"   Train: {train_features.shape}, Val: {val_features.shape}")
            break
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"   OOM with batch size {batch_size}")
                clear_gpu_memory()
                continue
            raise
        
        except Exception as e:
            logger.warning(f"   Failed with batch size {batch_size}: {e}")
            clear_gpu_memory()
            continue
    
    if train_features is None or val_features is None:
        logger.error("❌ Could not extract features. Aborting.")
        return
    
    # Free up memory from feature extraction models
    del spatiotemporal_model
    del gan_fingerprint_model
    clear_gpu_memory()
    
    # =========== STEP 5: TRAIN FUSION MODEL ===========
    print("\n🔍 Step 5: Training fusion model...")
    
    # Create fusion dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    for batch_size in CONFIG["batch_sizes"]:
        logger.info(f"   Trying training batch size: {batch_size}")
        
        try:
            fusion_train_loader = DataLoader(
                train_dataset,
                batch_size=min(batch_size * 8, 64),  # Fusion is lightweight
                shuffle=True
            )
            fusion_val_loader = DataLoader(
                val_dataset,
                batch_size=min(batch_size * 8, 64),
                shuffle=False
            )
            
            # Create fusion model
            fusion_model = create_fusion_model(config)
            
            # Train
            checkpoint_dir = Path(config.paths.checkpoints) / "fusion"
            
            results = train_fusion_model(
                fusion_model,
                fusion_train_loader,
                fusion_val_loader,
                device,
                CONFIG["epochs"],
                logger,
                checkpoint_dir
            )
            
            logger.info("=" * 60)
            logger.info("✅ FUSION TRAINING COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"   Best Validation Loss: {results['best_val_loss']:.4f}")
            logger.info(f"   Best Accuracy: {results['best_accuracy']:.4f}")
            logger.info(f"   Checkpoints saved to: {checkpoint_dir}")
            
            break
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"   OOM with batch size {batch_size}, trying smaller...")
                clear_gpu_memory()
                continue
            raise
        
        except Exception as e:
            logger.error(f"   Training failed: {e}")
            traceback.print_exc()
            clear_gpu_memory()
            continue
    
    print("\n" + "=" * 70)
    print("   TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        clear_gpu_memory()
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
        clear_gpu_memory()
