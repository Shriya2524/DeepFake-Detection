"""
Evaluation modules for deepfake detection.
"""

from .metrics import (
    compute_classification_metrics,
    compute_roc_auc,
    compute_confusion_matrix,
    EvaluationResults
)
from .visualize_results import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    plot_frame_scores,
    create_heatmap_overlay
)

__all__ = [
    "compute_classification_metrics",
    "compute_roc_auc",
    "compute_confusion_matrix",
    "EvaluationResults",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
    "plot_frame_scores",
    "create_heatmap_overlay",
]

