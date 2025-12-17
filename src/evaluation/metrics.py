"""
Evaluation Metrics Module

This module implements evaluation metrics as specified in Section 3.6:
- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1-score (macro and weighted)
- Confusion matrix
- Per-speaker accuracy
- Statistical significance tests
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'weighted', 'micro')
    
    Returns:
        Dictionary with all metrics
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        f'precision_{average}': precision,
        f'recall_{average}': recall,
        f'f1_{average}': f1
    }
    
    # Also calculate macro averages
    if average != 'macro':
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics.update({
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        })
    
    return metrics


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all metrics (both macro and weighted averages).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred)
    }
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics.update({
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    })
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics.update({
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    })
    
    return metrics


def per_speaker_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_speakers: Optional[int] = None
) -> Dict[int, float]:
    """
    Calculate accuracy for each speaker.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_speakers: Number of speakers (if None, inferred from data)
    
    Returns:
        Dictionary mapping speaker_id to accuracy
    """
    if num_speakers is None:
        num_speakers = max(max(y_true), max(y_pred)) + 1
    
    speaker_accuracies = {}
    
    for speaker_id in range(num_speakers):
        mask = y_true == speaker_id
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_true[mask])
            speaker_accuracies[speaker_id] = acc
        else:
            speaker_accuracies[speaker_id] = 0.0
    
    return speaker_accuracies


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', None)
    
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    
    return cm


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speaker_names: Optional[List[str]] = None
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        speaker_names: List of speaker names (optional)
    
    Returns:
        Classification report as string
    """
    target_names = speaker_names if speaker_names else None
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0
    )
    
    return report


def statistical_significance_test(
    scores_model1: np.ndarray,
    scores_model2: np.ndarray,
    test: str = 'wilcoxon',
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Perform statistical significance test to compare two models.
    
    Args:
        scores_model1: Accuracy scores for model 1 (per sample or per fold)
        scores_model2: Accuracy scores for model 2
        test: Test to use ('wilcoxon', 't_test_paired')
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    if len(scores_model1) != len(scores_model2):
        raise ValueError("Score arrays must have same length")
    
    if test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(scores_model1, scores_model2)
        test_name = "Wilcoxon signed-rank test"
    elif test == 't_test_paired':
        statistic, p_value = stats.ttest_rel(scores_model1, scores_model2)
        test_name = "Paired t-test"
    else:
        raise ValueError(f"Unknown test: {test}")
    
    is_significant = p_value < alpha
    
    mean_diff = np.mean(scores_model2 - scores_model1)
    
    result = {
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'is_significant': is_significant,
        'mean_difference': mean_diff,
        'interpretation': (
            f"Model 2 is {'significantly' if is_significant else 'not significantly'} "
            f"{'better' if mean_diff > 0 else 'worse'} than Model 1 (p={p_value:.4f})"
        )
    }
    
    return result


def compare_models(
    model1_predictions: np.ndarray,
    model2_predictions: np.ndarray,
    y_true: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> pd.DataFrame:
    """
    Compare two models across all metrics.
    
    Args:
        model1_predictions: Predictions from model 1
        model2_predictions: Predictions from model 2
        y_true: True labels
        model1_name: Name of model 1
        model2_name: Name of model 2
    
    Returns:
        DataFrame with comparison
    """
    metrics1 = calculate_all_metrics(y_true, model1_predictions)
    metrics2 = calculate_all_metrics(y_true, model2_predictions)
    
    comparison = []
    for metric_name in metrics1.keys():
        comparison.append({
            'Metric': metric_name,
            model1_name: metrics1[metric_name],
            model2_name: metrics2[metric_name],
            'Difference': metrics2[metric_name] - metrics1[metric_name]
        })
    
    df = pd.DataFrame(comparison)
    return df


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    speaker_names: Optional[List[str]] = None
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        speaker_names: List of speaker names
    
    Returns:
        Dictionary with all evaluation results
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test, y_pred)
    
    # Per-speaker accuracy
    speaker_acc = per_speaker_accuracy(y_test, y_pred)
    
    # Confusion matrix
    cm = get_confusion_matrix(y_test, y_pred)
    cm_normalized = get_confusion_matrix(y_test, y_pred, normalize='true')
    
    # Classification report
    report = print_classification_report(y_test, y_pred, speaker_names)
    
    results = {
        'metrics': metrics,
        'per_speaker_accuracy': speaker_acc,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    return results


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("=" * 50)
    print("\nMetrics implemented (Section 3.6):")
    print("- Accuracy")
    print("- Precision (macro, weighted)")
    print("- Recall (macro, weighted)")
    print("- F1-score (macro, weighted)")
    print("- Confusion Matrix")
    print("- Per-speaker Accuracy")
    print("- Statistical Significance Tests (Wilcoxon, t-test)")
