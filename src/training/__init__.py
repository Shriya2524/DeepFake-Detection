"""
Training modules for deepfake detection models.
"""

from .trainer import Trainer, TrainingConfig
from .losses import FocalLoss, LabelSmoothingLoss

__all__ = [
    "Trainer",
    "TrainingConfig",
    "FocalLoss",
    "LabelSmoothingLoss",
]

