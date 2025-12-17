"""
Training Callbacks Module

This module implements custom callbacks for training monitoring and control.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
import time


class TrainingLogger(keras.callbacks.Callback):
    """
    Custom callback to log training progress to file.
    """
    
    def __init__(self, log_dir: str = 'logs'):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save log files
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f'training_log_{int(time.time())}.json'
        self.epoch_logs = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch."""
        logs = logs or {}
        epoch_data = {
            'epoch': epoch + 1,
            'timestamp': time.time()
        }
        epoch_data.update(logs)
        self.epoch_logs.append(epoch_data)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.epoch_logs, f, indent=2)


class PerSpeakerAccuracy(keras.callbacks.Callback):
    """
    Custom callback to compute per-speaker accuracy during training.
    """
    
    def __init__(self, validation_data, num_speakers: int):
        """
        Initialize callback.
        
        Args:
            validation_data: Tuple of (X_val, y_val)
            num_speakers: Number of speakers
        """
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.num_speakers = num_speakers
    
    def on_epoch_end(self, epoch, logs=None):
        """Compute per-speaker accuracy at end of epoch."""
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(self.y_val, axis=1) if len(self.y_val.shape) > 1 else self.y_val
        
        # Compute accuracy for each speaker
        speaker_accuracies = []
        for speaker_id in range(self.num_speakers):
            mask = y_true_labels == speaker_id
            if np.sum(mask) > 0:
                acc = np.mean(y_pred_labels[mask] == y_true_labels[mask])
                speaker_accuracies.append(acc)
            else:
                speaker_accuracies.append(0.0)
        
        avg_per_speaker_acc = np.mean(speaker_accuracies)
        
        if logs is not None:
            logs['val_per_speaker_acc'] = avg_per_speaker_acc
        
        print(f"\nPer-speaker accuracy: {avg_per_speaker_acc:.4f}")


class LearningRateScheduler(keras.callbacks.Callback):
    """
    Custom learning rate scheduler with warmup and decay.
    """
    
    def __init__(
        self,
        initial_lr: float = 0.001,
        warmup_epochs: int = 5,
        decay_rate: float = 0.95,
        decay_epochs: int = 10
    ):
        """
        Initialize scheduler.
        
        Args:
            initial_lr: Initial learning rate
            warmup_epochs: Number of warmup epochs
            decay_rate: Learning rate decay rate
            decay_epochs: Apply decay every N epochs
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        """Update learning rate at beginning of epoch."""
        if epoch < self.warmup_epochs:
            # Warmup phase: linearly increase LR
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Decay phase
            decay_steps = (epoch - self.warmup_epochs) // self.decay_epochs
            lr = self.initial_lr * (self.decay_rate ** decay_steps)
        
        keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.6f}")


class ModelCheckpointWithMetadata(keras.callbacks.Callback):
    """
    Enhanced model checkpoint that saves metadata along with model.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_accuracy',
        save_best_only: bool = True,
        mode: str = 'max'
    ):
        """
        Initialize checkpoint.
        
        Args:
            filepath: Path to save model
            monitor: Metric to monitor
            save_best_only: Save only when monitored metric improves
            mode: 'min' or 'max'
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        
        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best = -float('inf')
            self.monitor_op = lambda x, y: x > y
    
    def on_epoch_end(self, epoch, logs=None):
        """Save model if metric improved."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if not self.save_best_only or self.monitor_op(current, self.best):
            # Save model
            self.model.save(self.filepath)
            
            # Save metadata
            metadata = {
                'epoch': epoch + 1,
                'best_' + self.monitor: float(current),
                'metrics': {k: float(v) for k, v in logs.items()}
            }
            
            metadata_path = Path(self.filepath).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if self.save_best_only:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model")
                self.best = current


if __name__ == "__main__":
    print("Training Callbacks Module")
    print("=" * 50)
    print("\nCustom callbacks available:")
    print("- TrainingLogger: Log training metrics to file")
    print("- PerSpeakerAccuracy: Track per-speaker accuracy")
    print("- LearningRateScheduler: Custom LR schedule with warmup")
    print("- ModelCheckpointWithMetadata: Save model with metadata")
