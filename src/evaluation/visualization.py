"""
Visualization Module

This module implements visualization functions for model evaluation and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_confusion_matrix(
    cm: np.ndarray,
    speaker_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        speaker_names: List of speaker names
        normalize: Whether to normalize
        title: Plot title
        cmap: Colormap
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if speaker_names is None:
        speaker_names = [f"Speaker {i}" for i in range(len(cm))]
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=speaker_names,
        yticklabels=speaker_names,
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_training_history(
    history: Dict,
    metrics: List[str] = ['accuracy', 'loss'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot training history (for CNN).
    
    Args:
        history: Training history dictionary
        metrics: List of metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training metric
        if metric in history:
            ax.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation metric
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    return fig


def plot_per_speaker_accuracy(
    speaker_accuracies: Dict[int, float],
    speaker_names: Optional[List[str]] = None,
    title: str = "Per-Speaker Accuracy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot per-speaker accuracy as bar chart.
    
    Args:
        speaker_accuracies: Dictionary mapping speaker_id to accuracy
        speaker_names: List of speaker names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    speaker_ids = sorted(speaker_accuracies.keys())
    accuracies = [speaker_accuracies[sid] for sid in speaker_ids]
    
    if speaker_names is None:
        speaker_names = [f"Speaker {sid}" for sid in speaker_ids]
    
    # Create bar plot
    bars = ax.bar(range(len(speaker_ids)), accuracies, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{accuracies[i]:.2f}',
                ha='center', va='bottom')
    
    # Add average line
    avg_acc = np.mean(accuracies)
    ax.axhline(y=avg_acc, color='red', linestyle='--', 
               label=f'Average: {avg_acc:.2f}')
    
    ax.set_xlabel('Speaker')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(range(len(speaker_ids)))
    ax.set_xticklabels(speaker_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-speaker accuracy saved to: {save_path}")
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot comparison between models.
    
    Args:
        comparison_df: DataFrame with comparison (from compare_models)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics = comparison_df['Metric'].values
    model_names = [col for col in comparison_df.columns if col not in ['Metric', 'Difference']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, model_name in enumerate(model_names):
        values = comparison_df[model_name].values
        offset = width * (i - len(model_names)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model_name, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    
    return fig


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    title: str = "Top Feature Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance (for Random Forest).
    
    Args:
        importance: Feature importance array
        feature_names: List of feature names
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top N features
    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_importance = importance[top_indices]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    top_names = [feature_names[i] for i in top_indices]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importance, color='steelblue', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance saved to: {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_speakers: int,
    speaker_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        num_speakers: Number of speakers
        speaker_names: List of speaker names
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(num_speakers))
    
    if speaker_names is None:
        speaker_names = [f"Speaker {i}" for i in range(num_speakers)]
    
    # Plot ROC curve for each speaker
    for i in range(num_speakers):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{speaker_names[i]} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Multi-class')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Visualization Module")
    print("=" * 50)
    print("\nVisualization functions available:")
    print("- plot_confusion_matrix()")
    print("- plot_training_history()")
    print("- plot_per_speaker_accuracy()")
    print("- plot_model_comparison()")
    print("- plot_feature_importance()")
    print("- plot_roc_curves()")
