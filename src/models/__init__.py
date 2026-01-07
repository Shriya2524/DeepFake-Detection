"""
Deep learning models for deepfake detection.

Includes:
- Spatiotemporal model for video analysis
- GAN fingerprint model for image analysis
- GAN family classifier for multi-class GAN source detection
- Fusion model for combining predictions
"""

from .backbones import get_backbone, BackboneType
from .spatiotemporal_model import SpatiotemporalModel, create_spatiotemporal_model
from .gan_fingerprint_model import GANFingerprintModel, create_gan_fingerprint_model
from .gan_family_model import (
    GANFamilyClassifier,
    create_gan_family_model,
    DEFAULT_GAN_FAMILY_CLASSES
)
from .fusion_model import FusionModel, create_fusion_model, DeepfakeDetector

__all__ = [
    "get_backbone",
    "BackboneType",
    "SpatiotemporalModel",
    "create_spatiotemporal_model",
    "GANFingerprintModel",
    "create_gan_fingerprint_model",
    "GANFamilyClassifier",
    "create_gan_family_model",
    "DEFAULT_GAN_FAMILY_CLASSES",
    "FusionModel",
    "create_fusion_model",
    "DeepfakeDetector",
]

