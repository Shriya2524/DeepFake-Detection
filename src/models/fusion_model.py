"""
Fusion model for combining spatiotemporal and GAN fingerprint predictions.

Takes predictions from both branches and produces a final deepfake probability.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatiotemporal_model import SpatiotemporalModel, create_spatiotemporal_model
from .gan_fingerprint_model import GANFingerprintModel, create_gan_fingerprint_model
from ..utils.logger import get_logger


logger = get_logger(__name__)


class FusionModel(nn.Module):
    """
    Fusion model that combines predictions from multiple branches.
    
    Takes aggregated predictions from:
    - Spatiotemporal model (video-level)
    - GAN fingerprint model (frame-level, aggregated)
    
    And produces a final deepfake probability.
    """
    
    def __init__(
        self,
        input_features: int = 5,
        hidden_dims: List[int] = [32, 16],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        """
        Initialize fusion model.
        
        Args:
            input_features: Number of input features
                Default 5: [p_video_fake, mean_p_image, std_p_image, max_p_image, min_p_image]
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__()
        
        layers = []
        in_dim = input_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(
            f"Created FusionModel: input_features={input_features}, "
            f"hidden_dims={hidden_dims}"
        )
    
    def forward(
        self,
        video_prob: torch.Tensor,
        image_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            video_prob: Video-level fake probability from spatiotemporal model (B,)
            image_probs: Frame-level fake probabilities from GAN fingerprint (B, T)
            
        Returns:
            Dictionary with:
            - logits: Classification logits (B, num_classes)
            - probs: Classification probabilities (B, num_classes)
            - fusion_features: Input features used for fusion (B, 5)
        """
        # Compute statistics from frame probabilities
        mean_prob = image_probs.mean(dim=1)
        std_prob = image_probs.std(dim=1)
        max_prob = image_probs.max(dim=1).values
        min_prob = image_probs.min(dim=1).values
        
        # Combine features
        features = torch.stack([
            video_prob,
            mean_prob,
            std_prob,
            max_prob,
            min_prob
        ], dim=1)
        
        # Classify
        logits = self.mlp(features)
        probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'fusion_features': features
        }
    
    def forward_features(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with pre-computed features.
        
        Args:
            features: Pre-computed fusion features (B, input_features)
            
        Returns:
            Dictionary with logits and probs
        """
        logits = self.mlp(features)
        probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs
        }


class DeepfakeDetector(nn.Module):
    """
    Complete deepfake detection system combining all models.
    
    This is the main inference class that:
    1. Runs spatiotemporal analysis on video
    2. Runs GAN fingerprint analysis on frames
    3. Fuses predictions for final result
    """
    
    def __init__(
        self,
        spatiotemporal_model: Optional[SpatiotemporalModel] = None,
        gan_fingerprint_model: Optional[GANFingerprintModel] = None,
        fusion_model: Optional[FusionModel] = None,
        config = None
    ):
        """
        Initialize complete detector.
        
        Args:
            spatiotemporal_model: Pre-initialized spatiotemporal model
            gan_fingerprint_model: Pre-initialized GAN fingerprint model
            fusion_model: Pre-initialized fusion model
            config: Configuration for creating models if not provided
        """
        super().__init__()
        
        # Initialize models
        if spatiotemporal_model is not None:
            self.spatiotemporal = spatiotemporal_model
        elif config is not None:
            self.spatiotemporal = create_spatiotemporal_model(config)
        else:
            self.spatiotemporal = SpatiotemporalModel()
        
        if gan_fingerprint_model is not None:
            self.gan_fingerprint = gan_fingerprint_model
        elif config is not None:
            self.gan_fingerprint = create_gan_fingerprint_model(config)
        else:
            self.gan_fingerprint = GANFingerprintModel()
        
        if fusion_model is not None:
            self.fusion = fusion_model
        elif config is not None:
            self.fusion = create_fusion_model(config)
        else:
            self.fusion = FusionModel()
        
        logger.info("Created complete DeepfakeDetector")
    
    def forward(
        self,
        frames: torch.Tensor,
        multichannel_frames: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all models.
        
        Args:
            frames: Video frames (B, T, C, H, W) - normalized RGB for spatiotemporal
            multichannel_frames: Multi-channel frames (B, T, 9, H, W) for GAN fingerprint
            flow: Optical flow (B, T-1, 2, H, W)
            mask: Valid frame mask (B, T)
            
        Returns:
            Dictionary with:
            - final_logits: Final classification logits (B, 2)
            - final_probs: Final classification probabilities (B, 2)
            - video_probs: Spatiotemporal model probabilities (B, 2)
            - frame_probs: Per-frame GAN fingerprint probabilities (B, T, 2)
            - frame_scores: Temporal attention scores (B, T)
        """
        B, T, C, H, W = frames.shape
        
        # Spatiotemporal analysis
        spatio_output = self.spatiotemporal(frames, flow, mask)
        video_probs = spatio_output['probs']
        video_fake_prob = video_probs[:, 1]  # Probability of fake
        
        # GAN fingerprint analysis on each frame
        mc_flat = multichannel_frames.view(B * T, -1, H, W)
        gan_output = self.gan_fingerprint(mc_flat)
        frame_probs = gan_output['probs'].view(B, T, -1)
        frame_fake_probs = frame_probs[:, :, 1]  # (B, T) - fake probability per frame
        
        # Mask invalid frames
        if mask is not None:
            frame_fake_probs = frame_fake_probs * mask.float()
        
        # Fusion
        fusion_output = self.fusion(video_fake_prob, frame_fake_probs)
        
        return {
            'final_logits': fusion_output['logits'],
            'final_probs': fusion_output['probs'],
            'video_probs': video_probs,
            'frame_probs': frame_probs,
            'frame_scores': spatio_output['frame_scores'],
            'fusion_features': fusion_output['fusion_features']
        }
    
    def predict(
        self,
        frames: torch.Tensor,
        multichannel_frames: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Make predictions with labels.
        
        Args:
            frames: Video frames (B, T, C, H, W)
            multichannel_frames: Multi-channel frames (B, T, 9, H, W)
            flow: Optical flow
            threshold: Classification threshold
            
        Returns:
            Dictionary with predictions and labels
        """
        output = self.forward(frames, multichannel_frames, flow)
        
        final_probs = output['final_probs']
        fake_prob = final_probs[:, 1]
        
        predictions = (fake_prob > threshold).long()
        labels = ['DEEPFAKE' if p == 1 else 'REAL' for p in predictions.tolist()]
        
        return {
            'predictions': predictions,
            'labels': labels,
            'fake_probability': fake_prob,
            'confidence': torch.abs(fake_prob - 0.5) * 2,  # Confidence based on distance from threshold
            'frame_scores': output['frame_scores']
        }
    
    def detect_video(
        self,
        frames: torch.Tensor,
        multichannel_frames: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[str, float, Dict]:
        """
        Detect if a single video is a deepfake.
        
        Convenience method for single-video inference.
        
        Args:
            frames: Video frames (1, T, C, H, W) or (T, C, H, W)
            multichannel_frames: Multi-channel frames
            flow: Optical flow
            threshold: Detection threshold
            
        Returns:
            Tuple of (label, probability, details)
        """
        # Add batch dimension if needed
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            multichannel_frames = multichannel_frames.unsqueeze(0)
            if flow is not None:
                flow = flow.unsqueeze(0)
        
        result = self.predict(frames, multichannel_frames, flow, threshold)
        
        label = result['labels'][0]
        prob = result['fake_probability'][0].item()
        
        details = {
            'confidence': result['confidence'][0].item(),
            'frame_scores': result['frame_scores'][0].detach().cpu().numpy()
        }
        
        return label, prob, details
    
    def load_pretrained(
        self,
        spatiotemporal_path: Optional[str] = None,
        gan_fingerprint_path: Optional[str] = None,
        fusion_path: Optional[str] = None
    ) -> None:
        """
        Load pretrained weights for each component.
        
        Args:
            spatiotemporal_path: Path to spatiotemporal model weights
            gan_fingerprint_path: Path to GAN fingerprint model weights
            fusion_path: Path to fusion model weights
        """
        if spatiotemporal_path:
            self.spatiotemporal.load_state_dict(
                torch.load(spatiotemporal_path, map_location='cpu')
            )
            logger.info(f"Loaded spatiotemporal weights from {spatiotemporal_path}")
        
        if gan_fingerprint_path:
            self.gan_fingerprint.load_state_dict(
                torch.load(gan_fingerprint_path, map_location='cpu')
            )
            logger.info(f"Loaded GAN fingerprint weights from {gan_fingerprint_path}")
        
        if fusion_path:
            self.fusion.load_state_dict(
                torch.load(fusion_path, map_location='cpu')
            )
            logger.info(f"Loaded fusion weights from {fusion_path}")


def create_fusion_model(config) -> FusionModel:
    """
    Create fusion model from configuration.
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        FusionModel instance
    """
    if hasattr(config, 'fusion_model'):
        cfg = config.fusion_model
    else:
        cfg = config.get('fusion_model', {})
    
    def get_value(obj, key, default):
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        return default
    
    hidden_dims = get_value(cfg, 'hidden_dims', [32, 16])
    if isinstance(hidden_dims, str):
        hidden_dims = eval(hidden_dims)
    
    return FusionModel(
        input_features=get_value(cfg, 'input_features', 5),
        hidden_dims=hidden_dims,
        dropout=get_value(cfg, 'dropout', 0.3),
        num_classes=get_value(cfg, 'num_classes', 2)
    )

