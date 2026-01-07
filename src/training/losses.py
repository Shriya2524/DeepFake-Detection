"""
Custom loss functions for deepfake detection training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the relative loss for well-classified examples,
    focusing training on hard negatives.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Target labels (B,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Helps prevent overconfident predictions and improves generalization.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 2
    ):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            smoothing: Label smoothing factor (0 = no smoothing)
            num_classes: Number of classes
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Target labels (B,)
            
        Returns:
            Loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    
    Can combine cross-entropy, focal loss, and auxiliary losses.
    """
    
    def __init__(
        self,
        main_loss: str = "cross_entropy",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        auxiliary_weight: float = 0.3
    ):
        """
        Initialize combined loss.
        
        Args:
            main_loss: Main loss type ("cross_entropy", "focal", "label_smoothing")
            focal_alpha: Alpha for focal loss
            focal_gamma: Gamma for focal loss
            label_smoothing: Label smoothing factor
            auxiliary_weight: Weight for auxiliary losses
        """
        super().__init__()
        
        if main_loss == "focal":
            self.main_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif main_loss == "label_smoothing":
            self.main_criterion = LabelSmoothingLoss(smoothing=label_smoothing)
        else:
            self.main_criterion = nn.CrossEntropyLoss()
        
        self.auxiliary_weight = auxiliary_weight
    
    def forward(
        self,
        main_output: torch.Tensor,
        targets: torch.Tensor,
        auxiliary_outputs: dict = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            main_output: Main prediction logits
            targets: Target labels
            auxiliary_outputs: Optional dictionary with auxiliary outputs
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Main loss
        main_loss = self.main_criterion(main_output, targets)
        losses['main'] = main_loss
        
        total_loss = main_loss
        
        # Auxiliary losses
        if auxiliary_outputs:
            if 'video_logits' in auxiliary_outputs:
                video_loss = self.main_criterion(auxiliary_outputs['video_logits'], targets)
                losses['video'] = video_loss
                total_loss = total_loss + self.auxiliary_weight * video_loss
            
            if 'frame_logits' in auxiliary_outputs:
                # Average frame loss
                frame_logits = auxiliary_outputs['frame_logits']  # (B, T, C)
                B, T, C = frame_logits.shape
                frame_targets = targets.unsqueeze(1).expand(B, T).reshape(-1)
                frame_loss = self.main_criterion(frame_logits.reshape(-1, C), frame_targets)
                losses['frame'] = frame_loss
                total_loss = total_loss + self.auxiliary_weight * frame_loss
        
        losses['total'] = total_loss
        
        return losses

