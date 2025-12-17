#!/usr/bin/env python
"""
Train CNN 1D Model for Speaker Identification

This script trains a 1D Convolutional Neural Network on sequential acoustic features.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import (
    load_config, set_random_seeds, setup_logging,
    save_results, print_system_info
)
from src.models.cnn_1d import CNN1DSpeakerClassifier
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_per_speaker_accuracy,
    plot_training_history
)
from src.data.dataset import FeatureDataset


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN 1D model for speaker identification'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--train-features',
        type=str,
        required=True,
        help='Path to training features file (.pkl or .h5)'
    )
    parser.add_argument(
        '--val-features',
        type=str,
        required=True,
        help='Path to validation features file'
    )
    parser.add_argument(
        '--test-features',
        type=str,
        default=None,
        help='Path to test features file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/cnn',
        help='Output directory for results'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/cnn_best.h5',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (-1 for CPU)'
    )
    
    args = parser.parse_args()
    
    # Setup GPU
    if args.gpu >= 0:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"Using GPU: {gpus[0]}")
            except RuntimeError as e:
                print(e)
    else:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Using CPU")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(
        log_level='INFO',
        log_file=output_dir / 'training.log'
    )
    
    logger.info("=" * 60)
    logger.info("CNN 1D Training Script")
    logger.info("=" * 60)
    
    # Print system info
    print_system_info()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set random seeds
    seed = args.seed if args.seed is not None else config.get('seeds', {}).get('tensorflow', 42)
    set_random_seeds(seed)
    
    # Load features
    logger.info("Loading training features...")
    if args.train_features.endswith('.pkl'):
        train_dataset = FeatureDataset.load_pickle(args.train_features)
    else:
        train_dataset = FeatureDataset.load_hdf5(args.train_features)
    
    X_train, y_train = train_dataset.get_data()
    logger.info(f"Training data: {X_train.shape}")
    
    logger.info("Loading validation features...")
    if args.val_features.endswith('.pkl'):
        val_dataset = FeatureDataset.load_pickle(args.val_features)
    else:
        val_dataset = FeatureDataset.load_hdf5(args.val_features)
    
    X_val, y_val = val_dataset.get_data()
    logger.info(f"Validation data: {X_val.shape}")
    
    # Get number of speakers and input shape
    num_speakers = len(np.unique(y_train))
    input_shape = X_train.shape[1:]  # (Tmax, F)
    logger.info(f"Number of speakers: {num_speakers}")
    logger.info(f"Input shape: {input_shape}")
    
    # Create model
    logger.info("Creating CNN 1D model...")
    model = CNN1DSpeakerClassifier(num_speakers, input_shape, config)
    model.build()
    
    # Create trainer
    trainer = ModelTrainer(model, config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        save_path=args.model_path
    )
    
    # Plot training history
    logger.info("Generating training history plots...")
    fig = plot_training_history(
        history,
        metrics=['accuracy', 'loss'],
        save_path=output_dir / 'training_history.png'
    )
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_results = evaluate_model(model, X_val, y_val)
    
    logger.info("\nValidation Results:")
    for metric, value in val_results['metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Plot confusion matrix
    logger.info("Generating confusion matrix...")
    fig = plot_confusion_matrix(
        val_results['confusion_matrix_normalized'],
        normalize=True,
        title='CNN 1D - Confusion Matrix (Validation)',
        save_path=output_dir / 'confusion_matrix_val.png'
    )
    
    # Plot per-speaker accuracy
    logger.info("Generating per-speaker accuracy plot...")
    fig = plot_per_speaker_accuracy(
        val_results['per_speaker_accuracy'],
        title='CNN 1D - Per-Speaker Accuracy (Validation)',
        save_path=output_dir / 'per_speaker_accuracy_val.png'
    )
    
    # Evaluate on test set if provided
    if args.test_features:
        logger.info("Loading test features...")
        if args.test_features.endswith('.pkl'):
            test_dataset = FeatureDataset.load_pickle(args.test_features)
        else:
            test_dataset = FeatureDataset.load_hdf5(args.test_features)
        
        X_test, y_test = test_dataset.get_data()
        logger.info(f"Test data: {X_test.shape}")
        
        logger.info("Evaluating on test set...")
        test_results = evaluate_model(model, X_test, y_test)
        
        logger.info("\nTest Results:")
        for metric, value in test_results['metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Plot test confusion matrix
        fig = plot_confusion_matrix(
            test_results['confusion_matrix_normalized'],
            normalize=True,
            title='CNN 1D - Confusion Matrix (Test)',
            save_path=output_dir / 'confusion_matrix_test.png'
        )
        
        # Save test results
        save_results(test_results['metrics'], output_dir / 'test_metrics.json')
    
    # Save validation results
    save_results(val_results['metrics'], output_dir / 'val_metrics.json')
    
    # Save training history
    save_results(history, output_dir / 'training_history.json')
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {args.model_path}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
