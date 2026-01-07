"""
Device management utilities for PyTorch.

Handles GPU/CPU selection and memory management.
"""

import os
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .logger import get_logger


logger = get_logger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device: Device specification
            - "auto": Automatically select GPU if available
            - "cuda": Use GPU (fails if not available)
            - "cuda:0", "cuda:1", etc.: Use specific GPU
            - "cpu": Use CPU
            
    Returns:
        torch.device object
        
    Raises:
        RuntimeError: If specified GPU is not available
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("GPU not available, using CPU")
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if ":" in device:
            gpu_id = int(device.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"GPU {gpu_id} not available. Available: {torch.cuda.device_count()}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info("Using CPU")
    
    return torch.device(device)


class DeviceManager:
    """
    Manages device placement and memory for PyTorch models.
    
    Example:
        manager = DeviceManager("auto")
        model = manager.prepare_model(model)
        data = manager.to_device(data)
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize device manager.
        
        Args:
            device: Device specification (auto, cuda, cuda:N, cpu)
        """
        self.device = get_device(device)
        self._memory_stats = {}
    
    def prepare_model(
        self,
        model: nn.Module,
        data_parallel: bool = False,
        gpu_ids: Optional[List[int]] = None
    ) -> nn.Module:
        """
        Prepare model for training/inference on the selected device.
        
        Args:
            model: PyTorch model
            data_parallel: Whether to use DataParallel for multi-GPU
            gpu_ids: List of GPU IDs for DataParallel
            
        Returns:
            Model on appropriate device
        """
        model = model.to(self.device)
        
        if data_parallel and self.device.type == "cuda":
            if gpu_ids is None:
                gpu_ids = list(range(torch.cuda.device_count()))
            
            if len(gpu_ids) > 1:
                model = nn.DataParallel(model, device_ids=gpu_ids)
                logger.info(f"Using DataParallel on GPUs: {gpu_ids}")
        
        return model
    
    def to_device(
        self,
        data: Union[torch.Tensor, dict, list, tuple],
        non_blocking: bool = True
    ) -> Union[torch.Tensor, dict, list, tuple]:
        """
        Move data to the managed device.
        
        Args:
            data: Tensor, dict of tensors, list of tensors, or tuple of tensors
            non_blocking: Whether to use non-blocking transfer
            
        Returns:
            Data on the managed device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=non_blocking)
        elif isinstance(data, dict):
            return {k: self.to_device(v, non_blocking) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            converted = [self.to_device(item, non_blocking) for item in data]
            return type(data)(converted)
        else:
            return data
    
    def get_memory_stats(self) -> dict:
        """
        Get current GPU memory statistics.
        
        Returns:
            Dictionary with memory stats (empty dict for CPU)
        """
        if self.device.type != "cuda":
            return {}
        
        return {
            "allocated": torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved(self.device) / 1024**3,    # GB
            "max_allocated": torch.cuda.max_memory_allocated(self.device) / 1024**3,
        }
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.debug("GPU memory cache cleared")
    
    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        stats = self.get_memory_stats()
        if stats:
            logger.info(
                f"GPU Memory: Allocated={stats['allocated']:.2f}GB, "
                f"Reserved={stats['reserved']:.2f}GB, "
                f"Max={stats['max_allocated']:.2f}GB"
            )


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: tuple,
    max_batch_size: int = 64,
    device: Optional[torch.device] = None
) -> int:
    """
    Find optimal batch size that fits in GPU memory.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        max_batch_size: Maximum batch size to try
        device: Device to test on
        
    Returns:
        Optimal batch size
    """
    if device is None:
        device = next(model.parameters()).device
    
    if device.type != "cuda":
        return max_batch_size
    
    model.eval()
    batch_size = max_batch_size
    
    while batch_size > 1:
        try:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            torch.cuda.empty_cache()
            logger.info(f"Optimal batch size: {batch_size}")
            return batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size //= 2
                torch.cuda.empty_cache()
            else:
                raise
    
    return 1

