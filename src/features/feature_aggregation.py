"""
Feature Aggregation Module

This module implements feature aggregation for Random Forest classifier.
Converts sequential features (T, F) to aggregated features (N,) by computing
statistical summaries (mean, std, min, max) over time dimension.

For 47 sequential features, produces 188 aggregated features (47 × 4).
"""

import numpy as np
from typing import Dict, List, Optional


def aggregate_features(
    features: np.ndarray,
    statistics: Optional[List[str]] = None
) -> np.ndarray:
    """
    Aggregate sequential features by computing statistics over time dimension.
    
    Args:
        features: Sequential features of shape (T, F) where T is time, F is features
        statistics: List of statistics to compute (default: ['mean', 'std', 'min', 'max'])
    
    Returns:
        Aggregated features of shape (F × len(statistics),)
    
    Example:
        >>> seq_features = np.random.rand(100, 47)  # 100 time frames, 47 features
        >>> agg_features = aggregate_features(seq_features)
        >>> print(agg_features.shape)  # (188,) = 47 features × 4 statistics
    """
    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max']
    
    aggregated = []
    
    for stat in statistics:
        if stat == 'mean':
            aggregated.append(np.mean(features, axis=0))
        elif stat == 'std':
            aggregated.append(np.std(features, axis=0))
        elif stat == 'min':
            aggregated.append(np.min(features, axis=0))
        elif stat == 'max':
            aggregated.append(np.max(features, axis=0))
        elif stat == 'median':
            aggregated.append(np.median(features, axis=0))
        elif stat == 'range':
            aggregated.append(np.ptp(features, axis=0))  # peak-to-peak
        elif stat == 'q25':
            aggregated.append(np.percentile(features, 25, axis=0))
        elif stat == 'q75':
            aggregated.append(np.percentile(features, 75, axis=0))
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    
    # Concatenate all statistics
    aggregated_features = np.concatenate(aggregated)
    
    return aggregated_features


def aggregate_features_dict(
    features_dict: Dict[str, np.ndarray],
    statistics: Optional[List[str]] = None
) -> np.ndarray:
    """
    Aggregate features from a dictionary (e.g., output of extract_all_features).
    
    Uses the 'sequential' key which contains combined features of shape (T, F).
    
    Args:
        features_dict: Dictionary with 'sequential' key containing features
        statistics: List of statistics to compute (default: ['mean', 'std', 'min', 'max'])
    
    Returns:
        Aggregated features of shape (F × len(statistics),)
    """
    if 'sequential' not in features_dict:
        raise ValueError("features_dict must contain 'sequential' key")
    
    sequential_features = features_dict['sequential']
    return aggregate_features(sequential_features, statistics)


def aggregate_feature_group(
    features: np.ndarray,
    feature_names: List[str],
    statistics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Aggregate features and return as a dictionary with descriptive names.
    
    Args:
        features: Sequential features of shape (T, F)
        feature_names: List of feature names (length F)
        statistics: List of statistics to compute
    
    Returns:
        Dictionary mapping feature_name_statistic to value
    
    Example:
        >>> features = np.random.rand(100, 2)
        >>> names = ['mfcc_1', 'mfcc_2']
        >>> agg = aggregate_feature_group(features, names)
        >>> print(agg.keys())  # ['mfcc_1_mean', 'mfcc_1_std', ..., 'mfcc_2_max']
    """
    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max']
    
    if len(feature_names) != features.shape[1]:
        raise ValueError("Number of feature names must match number of features")
    
    result = {}
    
    for i, feature_name in enumerate(feature_names):
        for stat in statistics:
            key = f"{feature_name}_{stat}"
            
            if stat == 'mean':
                value = np.mean(features[:, i])
            elif stat == 'std':
                value = np.std(features[:, i])
            elif stat == 'min':
                value = np.min(features[:, i])
            elif stat == 'max':
                value = np.max(features[:, i])
            elif stat == 'median':
                value = np.median(features[:, i])
            elif stat == 'range':
                value = np.ptp(features[:, i])
            elif stat == 'q25':
                value = np.percentile(features[:, i], 25)
            elif stat == 'q75':
                value = np.percentile(features[:, i], 75)
            else:
                raise ValueError(f"Unknown statistic: {stat}")
            
            result[key] = value
    
    return result


def get_feature_names(
    n_features: int = 47,
    statistics: Optional[List[str]] = None
) -> List[str]:
    """
    Generate feature names for aggregated features.
    
    Args:
        n_features: Number of sequential features (default: 47)
        statistics: List of statistics (default: ['mean', 'std', 'min', 'max'])
    
    Returns:
        List of feature names
    """
    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max']
    
    feature_names = []
    
    # MFCC features (40)
    for i in range(40):
        for stat in statistics:
            feature_names.append(f"mfcc_{i+1}_{stat}")
    
    # Pitch contour (1)
    for stat in statistics:
        feature_names.append(f"pitch_contour_{stat}")
    
    # Spectral features (3)
    for feature in ['spectral_centroid', 'spectral_rolloff', 'zcr']:
        for stat in statistics:
            feature_names.append(f"{feature}_{stat}")
    
    # Pitch statistics (3) - already aggregated but we compute stats anyway
    for feature in ['pitch_mean', 'pitch_std', 'pitch_range']:
        for stat in statistics:
            feature_names.append(f"{feature}_{stat}")
    
    return feature_names


def validate_aggregated_features(
    features: np.ndarray,
    expected_shape: int = 188
) -> bool:
    """
    Validate that aggregated features have the expected shape.
    
    Args:
        features: Aggregated features
        expected_shape: Expected number of features (default: 188)
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if features.ndim != 1:
        raise ValueError(f"Expected 1D array, got {features.ndim}D array")
    
    if features.shape[0] != expected_shape:
        raise ValueError(
            f"Expected {expected_shape} features, got {features.shape[0]}"
        )
    
    if np.isnan(features).any():
        raise ValueError("Features contain NaN values")
    
    if np.isinf(features).any():
        raise ValueError("Features contain infinite values")
    
    return True


if __name__ == "__main__":
    # Example usage
    print("Feature Aggregation Module")
    print("=" * 50)
    print("\nAggregation process:")
    print("- Input: Sequential features (T, 47)")
    print("- Statistics: mean, std, min, max")
    print("- Output: Aggregated features (188,) = 47 × 4")
    print("\nThis converts temporal features for use with Random Forest.")
    
    # Example
    print("\nExample:")
    seq_features = np.random.rand(100, 47)
    agg_features = aggregate_features(seq_features)
    print(f"Sequential features shape: {seq_features.shape}")
    print(f"Aggregated features shape: {agg_features.shape}")
    print(f"Valid: {validate_aggregated_features(agg_features)}")
