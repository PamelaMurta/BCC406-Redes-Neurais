"""
Helper Utilities Module

This module provides utility functions for configuration loading, logging, and common operations.
"""

import yaml
import json
import numpy as np
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {output_path}")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"Random seeds set to: {seed}")


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
    
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {log_level} level")
    
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def save_results(
    results: Dict,
    output_path: str,
    format: str = 'json'
) -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
        format: Format ('json', 'yaml')
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
    
    results_converted = convert_numpy(results)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
    elif format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(results_converted, f, default_flow_style=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved to: {output_path}")


def load_results(input_path: str) -> Dict:
    """
    Load results from file.
    
    Args:
        input_path: Path to results file
    
    Returns:
        Results dictionary
    """
    ext = Path(input_path).suffix.lower()
    
    if ext == '.json':
        with open(input_path, 'r') as f:
            results = json.load(f)
    elif ext in ['.yaml', '.yml']:
        with open(input_path, 'r') as f:
            results = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return results


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Project root path
    """
    return Path(__file__).parent.parent.parent


def print_system_info() -> None:
    """Print system information."""
    import platform
    import sys
    
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check for TensorFlow and GPU
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs available: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("No GPU available (using CPU)")
    except ImportError:
        print("TensorFlow not installed")
    
    # Check for other libraries
    try:
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
    except ImportError:
        pass
    
    try:
        import librosa
        print(f"librosa version: {librosa.__version__}")
    except ImportError:
        pass
    
    print("=" * 60)


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: Model instance (Keras or sklearn)
    
    Returns:
        Number of parameters
    """
    try:
        # For Keras models
        return model.count_params()
    except AttributeError:
        # For sklearn models
        try:
            return sum(tree.tree_.node_count for tree in model.estimators_)
        except:
            return 0


if __name__ == "__main__":
    print("Helper Utilities Module")
    print("=" * 50)
    print("\nUtility functions available:")
    print("- load_config(): Load configuration from YAML")
    print("- save_config(): Save configuration to YAML")
    print("- set_random_seeds(): Set seeds for reproducibility")
    print("- setup_logging(): Configure logging")
    print("- save_results(): Save results to JSON/YAML")
    print("- load_results(): Load results from file")
    print("- print_system_info(): Display system information")
    
    print("\nExample:")
    print_system_info()
