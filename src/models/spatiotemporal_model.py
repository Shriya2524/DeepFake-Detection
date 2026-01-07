"""
Spatiotemporal model for video-based deepfake detection.

Combines CNN feature extraction with temporal modeling (LSTM/GRU)
to detect temporal inconsistencies in deepfake videos.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_backbone, OpticalFlowEncoder
from ..utils.logger import get_logger


logger = get_logger(__name__)


class TemporalModel(nn.Module):
    """
    Temporal sequence model (LSTM, GRU, or Transformer).
    
    Processes sequence of frame features to capture temporal patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        model_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize temporal model.
        
        Args:
            input_dim: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of recurrent layers
            model_type: Type of model (lstm, gru, transformer)
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout probability
        """
        super().__init__()
        
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        
        if model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif model_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            )
            self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.bidirectional = False  # Transformer is inherently bidirectional
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Output dimension
        self.output_dim = hidden_size * (2 if bidirectional else 1)
        if model_type == "transformer":
            self.output_dim = input_dim
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence (B, T, D)
            mask: Optional mask for valid positions (B, T)
            
        Returns:
            Tuple of (sequence_output, final_output)
            - sequence_output: (B, T, output_dim)
            - final_output: (B, output_dim)
        """
        if self.model_type == "transformer":
            # Create attention mask from padding mask
            if mask is not None:
                attn_mask = ~mask  # Transformer uses inverse mask
            else:
                attn_mask = None
            
            output = self.rnn(x, src_key_padding_mask=attn_mask)
            
            # Pool over sequence
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).float()
                final = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                final = output.mean(dim=1)
                
        else:
            # LSTM or GRU
            output, hidden = self.rnn(x)
            
            if self.model_type == "lstm":
                # hidden is (h_n, c_n), we use h_n
                hidden = hidden[0]
            
            # Combine forward and backward hidden states
            if self.bidirectional:
                final = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                final = hidden[-1]
        
        return output, final
    
    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim


class SpatiotemporalModel(nn.Module):
    """
    Full spatiotemporal model for video deepfake detection.
    
    Architecture:
    1. CNN backbone extracts features from each frame
    2. Optional optical flow encoder processes motion information
    3. Temporal model (LSTM/GRU) captures temporal patterns
    4. Classification head produces final prediction
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        backbone_pretrained: bool = True,
        backbone_freeze_layers: int = 0,
        temporal_type: str = "lstm",
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_optical_flow: bool = True,
        flow_feature_dim: int = 128,
        num_classes: int = 2
    ):
        """
        Initialize spatiotemporal model.
        
        Args:
            backbone: CNN backbone type
            backbone_pretrained: Use pretrained backbone
            backbone_freeze_layers: Number of backbone layers to freeze
            temporal_type: Temporal model type (lstm, gru, transformer)
            hidden_size: Temporal model hidden size
            num_layers: Number of temporal layers
            bidirectional: Use bidirectional temporal model
            dropout: Dropout probability
            use_optical_flow: Whether to use optical flow features
            flow_feature_dim: Optical flow feature dimension
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.use_optical_flow = use_optical_flow
        
        # Frame feature extractor
        self.backbone = get_backbone(
            backbone_type=backbone,
            pretrained=backbone_pretrained,
            freeze_layers=backbone_freeze_layers
        )
        frame_feature_dim = self.backbone.get_feature_dim()
        
        # Optical flow encoder
        if use_optical_flow:
            self.flow_encoder = OpticalFlowEncoder(
                in_channels=2,
                feature_dim=flow_feature_dim
            )
            combined_dim = frame_feature_dim + flow_feature_dim
        else:
            self.flow_encoder = None
            combined_dim = frame_feature_dim
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Temporal model
        self.temporal_model = TemporalModel(
            input_dim=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            model_type=temporal_type,
            bidirectional=bidirectional,
            dropout=dropout
        )
        
        temporal_output_dim = self.temporal_model.get_output_dim()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_output_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Attention for frame-level importance (for explainability)
        self.frame_attention = nn.Sequential(
            nn.Linear(temporal_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        logger.info(
            f"Created SpatiotemporalModel: backbone={backbone}, "
            f"temporal={temporal_type}, hidden={hidden_size}"
        )
    
    def forward(
        self,
        frames: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: Video frames (B, T, C, H, W)
            flow: Optical flow (B, T-1, 2, H, W), optional
            mask: Valid frame mask (B, T), optional
            
        Returns:
            Dictionary with:
            - logits: Classification logits (B, num_classes)
            - probs: Classification probabilities (B, num_classes)
            - frame_scores: Per-frame anomaly scores (B, T)
            - features: Temporal features (B, temporal_dim)
        """
        B, T, C, H, W = frames.shape
        
        # Extract frame features
        frames_flat = frames.view(B * T, C, H, W)
        frame_features = self.backbone(frames_flat)
        frame_features = frame_features.view(B, T, -1)
        
        # Process optical flow if available
        if self.use_optical_flow and flow is not None and self.flow_encoder is not None:
            _, T_flow, C_flow, H_flow, W_flow = flow.shape
            flow_flat = flow.view(B * T_flow, C_flow, H_flow, W_flow)
            flow_features = self.flow_encoder(flow_flat)
            flow_features = flow_features.view(B, T_flow, -1)
            
            # Pad flow features to match frame length (T_flow = T - 1)
            padding = torch.zeros(B, 1, flow_features.shape[-1], device=flow.device)
            flow_features = torch.cat([flow_features, padding], dim=1)
            
            # Combine frame and flow features
            combined_features = torch.cat([frame_features, flow_features], dim=-1)
        else:
            combined_features = frame_features
        
        # Project features
        projected = self.feature_projection(combined_features)
        
        # Temporal modeling
        temporal_output, final_features = self.temporal_model(projected, mask)
        
        # Classification
        logits = self.classifier(final_features)
        probs = F.softmax(logits, dim=-1)
        
        # Frame attention scores (for explainability)
        attention_weights = self.frame_attention(temporal_output)
        attention_weights = attention_weights.squeeze(-1)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, float('-inf'))
        
        frame_scores = F.softmax(attention_weights, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'frame_scores': frame_scores,
            'features': final_features
        }
    
    def get_frame_anomaly_scores(
        self,
        frames: torch.Tensor,
        flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get per-frame anomaly scores.
        
        Useful for visualization and explanation.
        
        Args:
            frames: Video frames (B, T, C, H, W)
            flow: Optical flow (B, T-1, 2, H, W)
            
        Returns:
            Frame anomaly scores (B, T)
        """
        output = self.forward(frames, flow)
        
        # Weight frame scores by fake probability
        fake_prob = output['probs'][:, 1:2]  # (B, 1)
        weighted_scores = output['frame_scores'] * fake_prob
        
        return weighted_scores


def create_spatiotemporal_model(config) -> SpatiotemporalModel:
    """
    Create spatiotemporal model from configuration.
    
    Args:
        config: Configuration object or dictionary with model parameters
        
    Returns:
        SpatiotemporalModel instance
    """
    if hasattr(config, 'spatiotemporal_model'):
        cfg = config.spatiotemporal_model
    else:
        cfg = config.get('spatiotemporal_model', {})
    
    # Handle both Config objects and dicts
    def get_value(obj, key, default):
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        return default
    
    return SpatiotemporalModel(
        backbone=get_value(cfg, 'backbone', 'resnet18'),
        backbone_pretrained=get_value(cfg, 'backbone_pretrained', True),
        backbone_freeze_layers=get_value(cfg, 'backbone_freeze_layers', 0),
        temporal_type=get_value(cfg, 'temporal_type', 'lstm'),
        hidden_size=get_value(cfg, 'hidden_size', 256),
        num_layers=get_value(cfg, 'num_layers', 2),
        bidirectional=get_value(cfg, 'bidirectional', True),
        dropout=get_value(cfg, 'dropout', 0.3),
        use_optical_flow=get_value(cfg, 'use_optical_flow', True),
        num_classes=get_value(cfg, 'num_classes', 2)
    )

