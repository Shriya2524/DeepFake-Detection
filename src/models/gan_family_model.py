"""
GAN Family Classifier model for multi-class GAN source detection.

Identifies the source GAN architecture of generated images:
- Real (authentic media)
- StyleGAN (StyleGAN1, StyleGAN2, StyleGAN3)
- ProGAN (Progressive GAN)
- PGGAN (Progressive Growing GAN)
- Other (unknown GAN or other generators)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_backbone
from ..utils.logger import get_logger


logger = get_logger(__name__)


# Default GAN family class labels (order matters - must match training)
DEFAULT_GAN_FAMILY_CLASSES = [
    "Real",
    "StyleGAN",
    "ProGAN", 
    "PGGAN",
    "Other"
]


class GANFamilyClassifier(nn.Module):
    """
    Multi-class GAN family classifier.
    
    Identifies which GAN architecture generated an image, or if it's real.
    Uses similar architecture to GANFingerprintModel but with multi-class output.
    
    Architecture:
    1. Main CNN backbone processes RGB + color space channels
    2. Frequency branch analyzes DCT features for GAN-specific patterns
    3. Edge branch analyzes texture artifacts
    4. Features are fused and classified into GAN families
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        backbone_pretrained: bool = True,
        input_channels: int = 9,  # RGB(3) + CbCr(2) + DCT(1) + Edges(3)
        use_frequency_branch: bool = True,
        use_edge_branch: bool = True,
        feature_dim: int = 256,
        dropout: float = 0.4,
        num_classes: int = 5,  # Real, StyleGAN, ProGAN, PGGAN, Other
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize GAN family classifier.
        
        Args:
            backbone: CNN backbone type
            backbone_pretrained: Use pretrained backbone weights
            input_channels: Total input channels for main backbone
            use_frequency_branch: Include dedicated frequency branch
            use_edge_branch: Include dedicated edge/texture branch
            feature_dim: Final feature dimension before classification
            dropout: Dropout probability
            num_classes: Number of GAN family classes
            class_names: Optional list of class names (order must match training)
        """
        super().__init__()
        
        self.use_frequency_branch = use_frequency_branch
        self.use_edge_branch = use_edge_branch
        self.num_classes = num_classes
        self.class_names = class_names or DEFAULT_GAN_FAMILY_CLASSES[:num_classes]
        
        # Validate class names length matches num_classes
        if len(self.class_names) != num_classes:
            logger.warning(
                f"class_names length ({len(self.class_names)}) doesn't match "
                f"num_classes ({num_classes}). Padding/truncating."
            )
            if len(self.class_names) < num_classes:
                self.class_names = list(self.class_names) + [f"Class_{i}" for i in range(len(self.class_names), num_classes)]
            else:
                self.class_names = list(self.class_names)[:num_classes]
        
        # Channel indices for splitting multi-channel input
        self.rgb_channels = (0, 3)
        self.cbcr_channels = (3, 5)
        self.dct_channels = (5, 6)
        self.edge_channels = (6, 9)
        
        # Main backbone - processes RGB + color features
        main_input_channels = 5  # RGB + CbCr
        self.backbone = get_backbone(
            backbone_type=backbone,
            pretrained=backbone_pretrained,
            in_channels=main_input_channels
        )
        backbone_dim = self.backbone.get_feature_dim()
        
        total_feature_dim = backbone_dim
        
        # Frequency branch
        if use_frequency_branch:
            self.frequency_branch = FrequencyBranchFamily(
                in_channels=1,
                feature_dim=128
            )
            total_feature_dim += 128
        else:
            self.frequency_branch = None
        
        # Edge/texture branch
        if use_edge_branch:
            self.edge_branch = TextureEdgeBranchFamily(
                in_channels=3,
                feature_dim=128
            )
            total_feature_dim += 128
        else:
            self.edge_branch = None
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification head - multi-class
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        logger.info(
            f"Created GANFamilyClassifier: backbone={backbone}, "
            f"num_classes={num_classes}, classes={self.class_names}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Multi-channel input (B, C, H, W) where C=9
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with:
            - logits: Classification logits (B, num_classes)
            - probs: Classification probabilities (B, num_classes)
            - predicted_class: Index of predicted class (B,)
            - predicted_family: Name of predicted family (list of str)
            - features: Fused features (B, feature_dim) if return_features
        """
        # Split channels
        rgb = x[:, self.rgb_channels[0]:self.rgb_channels[1]]
        cbcr = x[:, self.cbcr_channels[0]:self.cbcr_channels[1]]
        
        # Main backbone input
        main_input = torch.cat([rgb, cbcr], dim=1)
        backbone_features = self.backbone(main_input)
        
        features_list = [backbone_features]
        
        # Frequency branch
        if self.use_frequency_branch and self.frequency_branch is not None:
            dct = x[:, self.dct_channels[0]:self.dct_channels[1]]
            freq_features = self.frequency_branch(dct)
            features_list.append(freq_features)
        
        # Edge branch
        if self.use_edge_branch and self.edge_branch is not None:
            edges = x[:, self.edge_channels[0]:self.edge_channels[1]]
            edge_features = self.edge_branch(edges)
            features_list.append(edge_features)
        
        # Fuse features
        combined = torch.cat(features_list, dim=1)
        fused = self.feature_fusion(combined)
        
        # Classify
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)
        
        # Map to family names
        predicted_family = [self.class_names[idx.item()] for idx in predicted_class]
        
        result = {
            'logits': logits,
            'probs': probs,
            'predicted_class': predicted_class,
            'predicted_family': predicted_family
        }
        
        if return_features:
            result['features'] = fused
            result['backbone_features'] = backbone_features
        
        return result
    
    def get_class_names(self) -> List[str]:
        """Return the list of class names."""
        return self.class_names
    
    def get_family_probs(self, probs: torch.Tensor) -> Dict[str, float]:
        """
        Convert probability tensor to family->probability mapping.
        
        Args:
            probs: Probability tensor (num_classes,) or (B, num_classes)
            
        Returns:
            Dictionary mapping family name to probability
        """
        if probs.dim() == 2:
            probs = probs[0]  # Take first sample
        
        return {
            name: probs[i].item()
            for i, name in enumerate(self.class_names)
        }


class FrequencyBranchFamily(nn.Module):
    """
    Specialized branch for processing frequency domain features.
    For GAN family classification.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 128
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
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
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, feature_dim)
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class TextureEdgeBranchFamily(nn.Module):
    """
    Branch for processing texture and edge features.
    For GAN family classification.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 128
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
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
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(128, feature_dim)
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def create_gan_family_model(
    config=None,
    class_names: Optional[List[str]] = None
) -> GANFamilyClassifier:
    """
    Create GAN family classifier from configuration.
    
    Args:
        config: Configuration object or dictionary
        class_names: Optional list of class names
        
    Returns:
        GANFamilyClassifier instance
    """
    if config is None:
        return GANFamilyClassifier(class_names=class_names)
    
    if hasattr(config, 'gan_family_model'):
        cfg = config.gan_family_model
    else:
        cfg = config.get('gan_family_model', {})
    
    def get_value(obj, key, default):
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        return default
    
    return GANFamilyClassifier(
        backbone=get_value(cfg, 'backbone', 'efficientnet_b0'),
        backbone_pretrained=get_value(cfg, 'backbone_pretrained', True),
        input_channels=get_value(cfg, 'input_channels', 9),
        use_frequency_branch=get_value(cfg, 'use_frequency', True),
        use_edge_branch=get_value(cfg, 'use_edge_features', True),
        feature_dim=get_value(cfg, 'feature_dim', 256),
        dropout=get_value(cfg, 'dropout', 0.4),
        num_classes=get_value(cfg, 'num_classes', 5),
        class_names=class_names or get_value(cfg, 'class_names', None)
    )
