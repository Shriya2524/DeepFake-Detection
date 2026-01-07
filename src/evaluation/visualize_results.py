"""
Visualization utilities for deepfake detection results.

Provides functions for plotting training curves, confusion matrices,
ROC curves, and frame-level analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from .metrics import compute_roc_auc, compute_precision_recall_curve


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Names for classes
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ["REAL", "DEEPFAKE"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Figure:
    """
    Plot ROC curve.
    
    Args:
        probabilities: Predicted probabilities
        targets: Ground truth labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    auc, fpr, tpr, _ = compute_roc_auc(probabilities, targets)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random')
    
    ax.fill_between(fpr, tpr, alpha=0.3, color='#2196F3')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Figure:
    """
    Plot precision-recall curve.
    
    Args:
        probabilities: Predicted probabilities
        targets: Ground truth labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = compute_precision_recall_curve(probabilities, targets)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='#4CAF50', lw=2, label='PR Curve')
    ax.fill_between(recall, precision, alpha=0.3, color='#4CAF50')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List],
    metrics: List[str] = None,
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with training history
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['loss', 'accuracy', 'f1']
    
    # Determine subplots layout
    n_plots = len(metrics)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Try to find the metric in history
        train_key = f'train_{metric}' if f'train_{metric}' in history else None
        val_key = f'val_{metric}' if f'val_{metric}' in history else None
        
        if metric == 'loss':
            train_key = 'train_loss'
            val_key = 'val_loss'
        
        # Plot training metric
        if train_key and train_key in history:
            train_values = history[train_key]
            ax.plot(epochs[:len(train_values)], train_values, 
                   label='Train', color='#2196F3', lw=2)
        
        # Plot validation metric
        if val_key and val_key in history:
            val_values = history[val_key]
            ax.plot(epochs[:len(val_values)], val_values, 
                   label='Validation', color='#FF5722', lw=2)
        
        # Try to extract from nested metrics
        if 'train_metrics' in history and 'val_metrics' in history:
            train_metrics = history['train_metrics']
            val_metrics = history['val_metrics']
            
            if train_metrics and metric in train_metrics[0]:
                train_values = [m.get(metric, 0) for m in train_metrics]
                ax.plot(epochs[:len(train_values)], train_values,
                       label='Train', color='#2196F3', lw=2)
            
            if val_metrics and metric in val_metrics[0]:
                val_values = [m.get(metric, 0) for m in val_metrics]
                ax.plot(epochs[:len(val_values)], val_values,
                       label='Validation', color='#FF5722', lw=2)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_title(f'{metric.upper()}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_frame_scores(
    frame_scores: np.ndarray,
    attention_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    title: str = "Frame-Level Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot frame-level scores for video analysis.
    
    Args:
        frame_scores: Per-frame fake probability scores
        attention_scores: Optional attention weights
        threshold: Detection threshold
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_frames = len(frame_scores)
    frames = np.arange(1, n_frames + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot frame scores
    colors = ['#4CAF50' if s < threshold else '#F44336' for s in frame_scores]
    bars = ax.bar(frames, frame_scores, color=colors, alpha=0.7, label='Frame Score')
    
    # Plot threshold line
    ax.axhline(y=threshold, color='#9E9E9E', linestyle='--', lw=2, label=f'Threshold ({threshold})')
    
    # Plot attention scores if available
    if attention_scores is not None:
        ax2 = ax.twinx()
        ax2.plot(frames, attention_scores, color='#FF9800', lw=2, 
                marker='o', markersize=3, label='Attention', alpha=0.8)
        ax2.set_ylabel('Attention Weight', fontsize=10, color='#FF9800')
        ax2.tick_params(axis='y', labelcolor='#FF9800')
        ax2.set_ylim(0, 1)
    
    ax.set_xlabel('Frame Number', fontsize=10)
    ax.set_ylabel('Fake Probability', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Legend
    handles1, labels1 = ax.get_legend_handles_labels()
    if attention_scores is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    else:
        ax.legend(loc='upper right')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay heatmap on image for visualization.
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Heatmap values (H, W) in [0, 1]
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
        
    Returns:
        Overlaid image (H, W, 3)
    """
    import cv2
    
    # Normalize heatmap to [0, 255]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Resize heatmap if needed
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def plot_detection_result(
    image: np.ndarray,
    label: str,
    probability: float,
    heatmap: Optional[np.ndarray] = None,
    title: str = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Plot detection result with optional heatmap overlay.
    
    Args:
        image: Input image
        label: Detection label (REAL/DEEPFAKE)
        probability: Detection probability
        heatmap: Optional attention/anomaly heatmap
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if heatmap is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=11)
        axes[0].axis('off')
        
        # Overlay
        overlay = create_heatmap_overlay(image, heatmap)
        axes[1].imshow(overlay)
        axes[1].set_title('Anomaly Heatmap', fontsize=11)
        axes[1].axis('off')
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        ax.axis('off')
    
    # Title with result
    color = '#F44336' if label == 'DEEPFAKE' else '#4CAF50'
    if title is None:
        title = f"Detection Result: {label} ({probability:.2%})"
    
    fig.suptitle(title, fontsize=14, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_evaluation_report(
    results: Dict,
    output_dir: Union[str, Path],
    prefix: str = "evaluation"
) -> None:
    """
    Save comprehensive evaluation report with all plots.
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save confusion matrix
    if 'confusion_matrix' in results:
        plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=output_dir / f"{prefix}_confusion_matrix.png"
        )
        plt.close()
    
    # Save ROC curve
    if 'probabilities' in results and 'labels' in results:
        plot_roc_curve(
            results['probabilities'],
            results['labels'],
            save_path=output_dir / f"{prefix}_roc_curve.png"
        )
        plt.close()
        
        plot_precision_recall_curve(
            results['probabilities'],
            results['labels'],
            save_path=output_dir / f"{prefix}_pr_curve.png"
        )
        plt.close()
    
    # Save training history
    if 'history' in results:
        plot_training_history(
            results['history'],
            save_path=output_dir / f"{prefix}_training_history.png"
        )
        plt.close()
    
    # Save metrics to text file
    metrics_path = output_dir / f"{prefix}_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
            elif key == 'classification_report':
                f.write(f"\n{value}\n")

