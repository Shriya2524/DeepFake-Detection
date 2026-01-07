"""
GAN Fingerprint model for image-based deepfake detection.

Analyzes images/frames for GAN-generated artifacts using:
- Multi-color space analysis
- Frequency domain features
- Edge and texture analysis
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_backbone
from ..utils.logger import get_logger


logger = get_logger(__name__)


class FrequencyBranch(nn.Module):
    """
    Specialized branch for processing frequency domain features.
    
    Learns to detect frequency patterns characteristic of GAN-generated images.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 128
    ):
        """
        Initialize frequency branch.
        
        Args:
            in_channels: Number of input channels (DCT, FFT features)
            feature_dim: Output feature dimension
        """
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
        """
        Forward pass.
        
        Args:
            x: Frequency features (B, C, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class TextureEdgeBranch(nn.Module):
    """
    Branch for processing texture and edge features.
    
    Analyzes Sobel and Laplacian filtered images to detect
    edge inconsistencies in deepfakes.
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # Sobel_x, Sobel_y, Laplacian
        feature_dim: int = 128
    ):
        """
        Initialize texture/edge branch.
        
        Args:
            in_channels: Number of input channels
            feature_dim: Output feature dimension
        """
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
        """Forward pass."""
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class GANFingerprintModel(nn.Module):
    """
    Full GAN fingerprint detection model.
    
    Architecture:
    1. Main CNN backbone processes RGB + color space channels
    2. Frequency branch analyzes DCT/FFT features
    3. Edge branch analyzes Sobel/Laplacian features
    4. Features are fused and classified
    
    This model is designed to detect subtle artifacts left by
    GAN generators in images.
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
        num_classes: int = 2
    ):
        """
        Initialize GAN fingerprint model.
        
        Args:
            backbone: CNN backbone type
            backbone_pretrained: Use pretrained backbone weights
            input_channels: Total input channels for main backbone
            use_frequency_branch: Include dedicated frequency branch
            use_edge_branch: Include dedicated edge/texture branch
            feature_dim: Final feature dimension before classification
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.use_frequency_branch = use_frequency_branch
        self.use_edge_branch = use_edge_branch
        
        # Channel indices for splitting multi-channel input
        # Assuming: RGB(0:3) + CbCr(3:5) + DCT(5:6) + Edges(6:9)
        self.rgb_channels = (0, 3)      # RGB
        self.cbcr_channels = (3, 5)     # Cb, Cr
        self.dct_channels = (5, 6)      # DCT high-freq
        self.edge_channels = (6, 9)     # Sobel_x, Sobel_y, Laplacian
        
        # Main backbone - processes RGB + color features
        main_input_channels = 5  # RGB + CbCr
        self.backbone = get_backbone(
            backbone_type=backbone,
            pretrained=backbone_pretrained,
            in_channels=main_input_channels
        )
        backbone_dim = self.backbone.get_feature_dim()
        
        # Frequency branch
        total_feature_dim = backbone_dim
        
        if use_frequency_branch:
            self.frequency_branch = FrequencyBranch(
                in_channels=1,  # DCT
                feature_dim=128
            )
            total_feature_dim += 128
        else:
            self.frequency_branch = None
        
        # Edge/texture branch
        if use_edge_branch:
            self.edge_branch = TextureEdgeBranch(
                in_channels=3,  # Sobel_x, Sobel_y, Laplacian
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
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        logger.info(
            f"Created GANFingerprintModel: backbone={backbone}, "
            f"freq_branch={use_frequency_branch}, edge_branch={use_edge_branch}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Multi-channel input (B, C, H, W) where C=9:
               - Channels 0-2: RGB
               - Channels 3-4: Cb, Cr
               - Channel 5: DCT high-frequency
               - Channels 6-8: Sobel_x, Sobel_y, Laplacian
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with:
            - logits: Classification logits (B, num_classes)
            - probs: Classification probabilities (B, num_classes)
            - features: Fused features (B, feature_dim) if return_features
        """
        # Split channels
        rgb = x[:, self.rgb_channels[0]:self.rgb_channels[1]]
        cbcr = x[:, self.cbcr_channels[0]:self.cbcr_channels[1]]
        
        # Main backbone input: RGB + CbCr
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
        
        result = {
            'logits': logits,
            'probs': probs
        }
        
        if return_features:
            result['features'] = fused
            result['backbone_features'] = backbone_features
        
        return result
    
    def forward_rgb_only(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with RGB input only (without multi-channel features).
        
        Creates dummy channels for other features. Useful for quick inference.
        
        Args:
            rgb: RGB input (B, 3, H, W)
            
        Returns:
            Same as forward()
        """
        B, _, H, W = rgb.shape
        device = rgb.device
        
        # Create dummy channels
        dummy_cbcr = torch.zeros(B, 2, H, W, device=device)
        dummy_dct = torch.zeros(B, 1, H, W, device=device)
        dummy_edges = torch.zeros(B, 3, H, W, device=device)
        
        x = torch.cat([rgb, dummy_cbcr, dummy_dct, dummy_edges], dim=1)
        
        return self.forward(x)


class GANFingerprintModelSimple(nn.Module):
    """
    Simplified GAN fingerprint model using only RGB input.
    
    For cases where multi-channel preprocessing is not available.
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        backbone_pretrained: bool = True,
        feature_dim: int = 256,
        dropout: float = 0.4,
        num_classes: int = 2
    ):
        """
        Initialize simplified model.
        
        Args:
            backbone: CNN backbone type
            backbone_pretrained: Use pretrained weights
            feature_dim: Feature dimension
            dropout: Dropout probability
            num_classes: Number of classes
        """
        super().__init__()
        
        self.backbone = get_backbone(
            backbone_type=backbone,
            pretrained=backbone_pretrained,
            in_channels=3
        )
        
        backbone_dim = self.backbone.get_feature_dim()
        
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'features': features
        }


def create_gan_fingerprint_model(config) -> GANFingerprintModel:
    """
    Create GAN fingerprint model from configuration.
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        GANFingerprintModel instance
    """
    if hasattr(config, 'gan_fingerprint_model'):
        cfg = config.gan_fingerprint_model
    else:
        cfg = config.get('gan_fingerprint_model', {})
    
    def get_value(obj, key, default):
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        return default
    
    return GANFingerprintModel(
        backbone=get_value(cfg, 'backbone', 'efficientnet_b0'),
        backbone_pretrained=get_value(cfg, 'backbone_pretrained', True),
        input_channels=get_value(cfg, 'input_channels', 9),
        use_frequency_branch=get_value(cfg, 'use_frequency', True),
        use_edge_branch=get_value(cfg, 'use_edge_features', True),
        feature_dim=get_value(cfg, 'feature_dim', 256),
        dropout=get_value(cfg, 'dropout', 0.4),
        num_classes=get_value(cfg, 'num_classes', 2)
    )

