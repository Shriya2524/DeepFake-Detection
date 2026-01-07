"""
Evaluation metrics for deepfake detection.

Provides functions for computing classification metrics,
ROC-AUC, confusion matrices, and comprehensive evaluation reports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    predictions: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    threshold: float = 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'threshold': self.threshold
        }
    
    def __repr__(self) -> str:
        return (
            f"EvaluationResults(\n"
            f"  accuracy={self.accuracy:.4f},\n"
            f"  precision={self.precision:.4f},\n"
            f"  recall={self.recall:.4f},\n"
            f"  f1={self.f1:.4f},\n"
            f"  roc_auc={self.roc_auc:.4f}\n"
            f")"
        )


def to_numpy(x: Union[torch.Tensor, np.ndarray, List]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    return x


def get_predictions(
    logits_or_probs: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert logits or probabilities to predictions.
    
    Args:
        logits_or_probs: Model outputs (B, num_classes) or (B,)
        threshold: Classification threshold
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    arr = to_numpy(logits_or_probs)
    
    if arr.ndim == 1:
        # Already probabilities for positive class
        probs = arr
    elif arr.shape[1] == 2:
        # Two-class output, take positive class probability
        if arr.max() > 1 or arr.min() < 0:
            # Logits, apply softmax
            exp_arr = np.exp(arr - arr.max(axis=1, keepdims=True))
            probs = exp_arr[:, 1] / exp_arr.sum(axis=1)
        else:
            probs = arr[:, 1]
    else:
        # Multi-class, use argmax
        probs = arr
        predictions = arr.argmax(axis=1)
        return predictions, probs
    
    predictions = (probs > threshold).astype(int)
    return predictions, probs


def compute_classification_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions (logits, probs, or class indices)
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary with metric values
    """
    preds, probs = get_predictions(predictions, threshold)
    targets = to_numpy(targets)
    
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, zero_division=0)
    }
    
    # ROC-AUC (if we have probabilities)
    if probs.ndim == 1 and len(np.unique(targets)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(targets, probs)
        except ValueError:
            metrics['roc_auc'] = 0.5
    
    return metrics


def compute_roc_auc(
    probabilities: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC-AUC and curve data.
    
    Args:
        probabilities: Predicted probabilities
        targets: Ground truth labels
        
    Returns:
        Tuple of (auc_score, fpr, tpr, thresholds)
    """
    probs = to_numpy(probabilities)
    targets = to_numpy(targets)
    
    if probs.ndim == 2:
        probs = probs[:, 1]
    
    try:
        fpr, tpr, thresholds = roc_curve(targets, probs)
        auc = roc_auc_score(targets, probs)
    except ValueError:
        fpr = np.array([0, 1])
        tpr = np.array([0, 1])
        thresholds = np.array([1, 0])
        auc = 0.5
    
    return auc, fpr, tpr, thresholds


def compute_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Confusion matrix (2x2 for binary classification)
    """
    preds, _ = get_predictions(predictions, threshold)
    targets = to_numpy(targets)
    
    return confusion_matrix(targets, preds)


def compute_precision_recall_curve(
    probabilities: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.
    
    Args:
        probabilities: Predicted probabilities
        targets: Ground truth labels
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    probs = to_numpy(probabilities)
    targets = to_numpy(targets)
    
    if probs.ndim == 2:
        probs = probs[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    
    return precision, recall, thresholds


def find_optimal_threshold(
    probabilities: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        probabilities: Predicted probabilities
        targets: Ground truth labels
        metric: Metric to optimize ("f1", "accuracy", "youden")
        
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    probs = to_numpy(probabilities)
    targets = to_numpy(targets)
    
    if probs.ndim == 2:
        probs = probs[:, 1]
    
    # Try different thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        
        if metric == "f1":
            score = f1_score(targets, preds, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(targets, preds)
        elif metric == "youden":
            # Youden's J statistic = TPR + TNR - 1
            tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = tpr + tnr - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def evaluate_model(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    class_names: List[str] = None
) -> EvaluationResults:
    """
    Comprehensive model evaluation.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        threshold: Classification threshold
        class_names: Names for classes
        
    Returns:
        EvaluationResults object with all metrics
    """
    if class_names is None:
        class_names = ["REAL", "DEEPFAKE"]
    
    preds, probs = get_predictions(predictions, threshold)
    targets = to_numpy(targets)
    
    # Compute metrics
    metrics = compute_classification_metrics(predictions, targets, threshold)
    cm = compute_confusion_matrix(predictions, targets, threshold)
    
    # Classification report
    report = classification_report(
        targets,
        preds,
        target_names=class_names,
        zero_division=0
    )
    
    return EvaluationResults(
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        roc_auc=metrics.get('roc_auc', 0.5),
        confusion_matrix=cm,
        classification_report=report,
        predictions=preds,
        probabilities=probs,
        labels=targets,
        threshold=threshold
    )


def compute_video_level_metrics(
    frame_predictions: List[np.ndarray],
    video_labels: List[int],
    aggregation: str = "mean"
) -> Dict[str, float]:
    """
    Compute video-level metrics from frame predictions.
    
    Args:
        frame_predictions: List of per-frame probability arrays
        video_labels: Video-level labels
        aggregation: How to aggregate frame predictions ("mean", "max", "voting")
        
    Returns:
        Dictionary with video-level metrics
    """
    video_probs = []
    
    for frame_probs in frame_predictions:
        if aggregation == "mean":
            video_prob = frame_probs.mean()
        elif aggregation == "max":
            video_prob = frame_probs.max()
        elif aggregation == "voting":
            video_prob = (frame_probs > 0.5).mean()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        video_probs.append(video_prob)
    
    video_probs = np.array(video_probs)
    video_labels = np.array(video_labels)
    
    return compute_classification_metrics(video_probs, video_labels)

