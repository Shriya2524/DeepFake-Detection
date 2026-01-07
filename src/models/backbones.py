"""
CNN backbone models for feature extraction.

Provides pretrained backbones from torchvision with customizable
feature dimensions and layer freezing.
"""

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
)

from ..utils.logger import get_logger


logger = get_logger(__name__)


class BackboneType(Enum):
    """Available backbone architectures."""
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"


class BackboneWrapper(nn.Module):
    """
    Wrapper for CNN backbones that outputs features instead of classification.
    
    Removes the final classification layer and optionally freezes early layers.
    """
    
    def __init__(
        self,
        backbone_type: str = "resnet18",
        pretrained: bool = True,
        freeze_layers: int = 0,
        in_channels: int = 3
    ):
        """
        Initialize backbone wrapper.
        
        Args:
            backbone_type: Type of backbone (resnet18, efficientnet_b0, etc.)
            pretrained: Whether to use pretrained weights
            freeze_layers: Number of layers to freeze (0 = none)
            in_channels: Number of input channels (modify first conv if != 3)
        """
        super().__init__()
        
        self.backbone_type = backbone_type
        self.in_channels = in_channels
        
        # Load backbone
        self.backbone, self.feature_dim = self._load_backbone(
            backbone_type, pretrained
        )
        
        # Modify first conv if needed
        if in_channels != 3:
            self._modify_first_conv(in_channels)
        
        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _load_backbone(
        self,
        backbone_type: str,
        pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """Load backbone model and return it with feature dimension."""
        
        if backbone_type == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif backbone_type == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet34(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif backbone_type == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif backbone_type == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        elif backbone_type == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b1(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        logger.info(f"Loaded backbone: {backbone_type} (feature_dim={feature_dim})")
        
        return model, feature_dim
    
    def _modify_first_conv(self, in_channels: int) -> None:
        """Modify the first convolutional layer for different input channels."""
        
        if self.backbone_type.startswith("resnet"):
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            # Initialize with pretrained weights for first 3 channels
            if in_channels >= 3:
                with torch.no_grad():
                    self.backbone.conv1.weight[:, :3] = old_conv.weight
                    if in_channels > 3:
                        # Initialize extra channels with mean of RGB weights
                        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                        self.backbone.conv1.weight[:, 3:] = mean_weight.repeat(
                            1, in_channels - 3, 1, 1
                        )
                        
        elif self.backbone_type.startswith("efficientnet"):
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            if in_channels >= 3:
                with torch.no_grad():
                    self.backbone.features[0][0].weight[:, :3] = old_conv.weight
                    if in_channels > 3:
                        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                        self.backbone.features[0][0].weight[:, 3:] = mean_weight.repeat(
                            1, in_channels - 3, 1, 1
                        )
        
        logger.info(f"Modified first conv for {in_channels} input channels")
    
    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the first N layers of the backbone."""
        
        if self.backbone_type.startswith("resnet"):
            # ResNet has: conv1, bn1, layer1, layer2, layer3, layer4
            layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
            layers_to_freeze = layers_to_freeze[:min(num_layers, len(layers_to_freeze))]
            
            for name, param in self.backbone.named_parameters():
                if any(name.startswith(layer) for layer in layers_to_freeze):
                    param.requires_grad = False
                    
        elif self.backbone_type.startswith("efficientnet"):
            # EfficientNet has features[0] through features[8]
            for i, block in enumerate(self.backbone.features):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
        
        frozen_count = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total_count = sum(1 for p in self.backbone.parameters())
        logger.info(f"Froze {frozen_count}/{total_count} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        return self.backbone(x)
    
    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim


def get_backbone(
    backbone_type: str = "resnet18",
    pretrained: bool = True,
    freeze_layers: int = 0,
    in_channels: int = 3
) -> BackboneWrapper:
    """
    Create a backbone feature extractor.
    
    Args:
        backbone_type: Type of backbone architecture
        pretrained: Whether to use pretrained weights
        freeze_layers: Number of layers to freeze
        in_channels: Number of input channels
        
    Returns:
        BackboneWrapper instance
    """
    return BackboneWrapper(
        backbone_type=backbone_type,
        pretrained=pretrained,
        freeze_layers=freeze_layers,
        in_channels=in_channels
    )


class OpticalFlowEncoder(nn.Module):
    """
    Simple CNN encoder for optical flow maps.
    
    Takes optical flow (2 channels: x, y) and produces feature embeddings.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        feature_dim: int = 128
    ):
        """
        Initialize optical flow encoder.
        
        Args:
            in_channels: Number of flow channels (typically 2 for x, y)
            feature_dim: Output feature dimension
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Optical flow tensor (B, 2, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        return self.encoder(x)

