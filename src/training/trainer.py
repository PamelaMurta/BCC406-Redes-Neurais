"""
Model Trainer Module

This module implements a generic trainer for both Random Forest and CNN models.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
from pathlib import Path


class ModelTrainer:
    """
    Generic trainer for speaker identification models.
    
    Handles training, validation, and saving of models.
    """
    
    def __init__(
        self,
        model,
        config: Optional[Dict] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model instance (RandomForestSpeakerClassifier or CNN1DSpeakerClassifier)
            config: Configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self.training_history = {}
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            save_path: Path to save trained model (optional)
        
        Returns:
            Training history/results
        """
        print("\n" + "=" * 60)
        print(f"Training {self.model.__class__.__name__}")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val) if X_val is not None else 0}")
        print(f"Number of speakers: {self.model.num_speakers}")
        print(f"Feature shape: {X_train.shape}")
        print("=" * 60)
        
        # Start timer
        start_time = time.time()
        
        # Train model
        history = self.model.train(X_train, y_train, X_val, y_val)
        
        # End timer
        elapsed_time = time.time() - start_time
        
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        # Save training history
        self.training_history = history
        
        # Save model if path provided
        if save_path is not None:
            self._save_model(save_path)
        
        return history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        print(f"Test samples: {len(X_test)}")
        
        metrics = self.model.evaluate(X_test, y_test)
        
        print("\nTest Results:")
        for metric_name, value in metrics.items():
            print(f"  - {metric_name}: {value:.4f}")
        
        return metrics
    
    def _save_model(self, save_path: str) -> None:
        """
        Save trained model.
        
        Args:
            save_path: Path to save model
        """
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(save_path)
        print(f"\nModel saved to: {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load trained model.
        
        Args:
            load_path: Path to load model from
        """
        self.model.load(load_path)
        print(f"Model loaded from: {load_path}")
    
    def get_training_history(self) -> Dict:
        """
        Get training history.
        
        Returns:
            Training history dictionary
        """
        return self.training_history


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_speakers: int,
    config: Dict,
    save_path: str
) -> Tuple[Any, Dict]:
    """
    Helper function to train Random Forest model.
    
    Args:
        X_train: Training features (N, 188)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        num_speakers: Number of speakers
        config: Configuration dictionary
        save_path: Path to save model
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    from .random_forest import RandomForestSpeakerClassifier
    
    # Create model
    model = RandomForestSpeakerClassifier(num_speakers, config)
    model.build()
    
    # Create trainer
    trainer = ModelTrainer(model, config)
    
    # Train model
    history = trainer.train(X_train, y_train, X_val, y_val, save_path)
    
    return model, history


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_speakers: int,
    config: Dict,
    save_path: str,
    input_shape: Tuple[int, int] = (100, 47)
) -> Tuple[Any, Dict]:
    """
    Helper function to train CNN model.
    
    Args:
        X_train: Training features (N, Tmax, F)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        num_speakers: Number of speakers
        config: Configuration dictionary
        save_path: Path to save model
        input_shape: Input shape (Tmax, F)
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    from .cnn_1d import CNN1DSpeakerClassifier
    
    # Create model
    model = CNN1DSpeakerClassifier(num_speakers, input_shape, config)
    model.build()
    
    # Create trainer
    trainer = ModelTrainer(model, config)
    
    # Train model
    history = trainer.train(X_train, y_train, X_val, y_val, save_path)
    
    return model, history


if __name__ == "__main__":
    print("Model Trainer Module")
    print("=" * 50)
    print("\nProvides generic training interface for all models.")
    print("Supports both Random Forest and CNN training.")
