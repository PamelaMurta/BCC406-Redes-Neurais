#!/usr/bin/env python
"""
Evaluate and Compare Speaker Identification Models

This script evaluates trained models and performs comparative analysis.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import (
    load_config, setup_logging, save_results, print_system_info
)
from src.models.random_forest import RandomForestSpeakerClassifier
from src.models.cnn_1d import CNN1DSpeakerClassifier
from src.evaluation.metrics import (
    evaluate_model, compare_models, statistical_significance_test,
    per_speaker_accuracy
)
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_per_speaker_accuracy,
    plot_model_comparison
)
from src.data.dataset import FeatureDataset


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare speaker identification models'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--rf-model',
        type=str,
        required=True,
        help='Path to Random Forest model (.pkl)'
    )
    parser.add_argument(
        '--cnn-model',
        type=str,
        required=True,
        help='Path to CNN model (.h5)'
    )
    parser.add_argument(
        '--test-features-rf',
        type=str,
        required=True,
        help='Path to test features for RF (aggregated)'
    )
    parser.add_argument(
        '--test-features-cnn',
        type=str,
        required=True,
        help='Path to test features for CNN (sequential)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/comparison',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(
        log_level='INFO',
        log_file=output_dir / 'evaluation.log'
    )
    
    logger.info("=" * 60)
    logger.info("Model Evaluation and Comparison Script")
    logger.info("=" * 60)
    
    # Print system info
    print_system_info()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Load test features for RF
    logger.info("Loading test features for Random Forest...")
    if args.test_features_rf.endswith('.pkl'):
        rf_test_dataset = FeatureDataset.load_pickle(args.test_features_rf)
    else:
        rf_test_dataset = FeatureDataset.load_hdf5(args.test_features_rf)
    
    X_test_rf, y_test = rf_test_dataset.get_data()
    logger.info(f"RF test data: {X_test_rf.shape}")
    
    # Load test features for CNN
    logger.info("Loading test features for CNN...")
    if args.test_features_cnn.endswith('.pkl'):
        cnn_test_dataset = FeatureDataset.load_pickle(args.test_features_cnn)
    else:
        cnn_test_dataset = FeatureDataset.load_hdf5(args.test_features_cnn)
    
    X_test_cnn, y_test_cnn = cnn_test_dataset.get_data()
    logger.info(f"CNN test data: {X_test_cnn.shape}")
    
    # Verify labels match
    if not np.array_equal(y_test, y_test_cnn):
        logger.warning("Test labels don't match between RF and CNN datasets!")
    
    num_speakers = len(np.unique(y_test))
    logger.info(f"Number of speakers: {num_speakers}")
    
    # Load Random Forest model
    logger.info(f"Loading Random Forest model from: {args.rf_model}")
    rf_model = RandomForestSpeakerClassifier(num_speakers, config)
    rf_model.load(args.rf_model)
    
    # Load CNN model
    logger.info(f"Loading CNN model from: {args.cnn_model}")
    input_shape = X_test_cnn.shape[1:]
    cnn_model = CNN1DSpeakerClassifier(num_speakers, input_shape, config)
    cnn_model.load(args.cnn_model)
    
    # Evaluate Random Forest
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Random Forest Model")
    logger.info("=" * 60)
    rf_results = evaluate_model(rf_model, X_test_rf, y_test)
    
    logger.info("\nRandom Forest Test Results:")
    for metric, value in rf_results['metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Evaluate CNN
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating CNN Model")
    logger.info("=" * 60)
    cnn_results = evaluate_model(cnn_model, X_test_cnn, y_test_cnn)
    
    logger.info("\nCNN Test Results:")
    for metric, value in cnn_results['metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Compare models
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison")
    logger.info("=" * 60)
    
    comparison_df = compare_models(
        rf_results['predictions'],
        cnn_results['predictions'],
        y_test,
        model1_name="Random Forest",
        model2_name="CNN 1D"
    )
    
    logger.info("\nComparison Table:")
    logger.info("\n" + comparison_df.to_string())
    
    # Save comparison table
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Plot model comparison
    logger.info("\nGenerating comparison plots...")
    fig = plot_model_comparison(
        comparison_df,
        title='Random Forest vs CNN 1D - Performance Comparison',
        save_path=output_dir / 'model_comparison.png'
    )
    
    # Plot confusion matrices side by side
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RF confusion matrix
    import seaborn as sns
    sns.heatmap(
        rf_results['confusion_matrix_normalized'],
        annot=True, fmt='.2f', cmap='Blues',
        ax=axes[0], cbar_kws={'label': 'Proportion'}
    )
    axes[0].set_title('Random Forest - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # CNN confusion matrix
    sns.heatmap(
        cnn_results['confusion_matrix_normalized'],
        annot=True, fmt='.2f', cmap='Blues',
        ax=axes[1], cbar_kws={'label': 'Proportion'}
    )
    axes[1].set_title('CNN 1D - Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrices saved to: {output_dir / 'confusion_matrices_comparison.png'}")
    
    # Per-speaker accuracy comparison
    rf_per_speaker = rf_results['per_speaker_accuracy']
    cnn_per_speaker = cnn_results['per_speaker_accuracy']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_speakers)
    width = 0.35
    
    rf_accs = [rf_per_speaker[i] for i in range(num_speakers)]
    cnn_accs = [cnn_per_speaker[i] for i in range(num_speakers)]
    
    ax.bar(x - width/2, rf_accs, width, label='Random Forest', alpha=0.7)
    ax.bar(x + width/2, cnn_accs, width, label='CNN 1D', alpha=0.7)
    
    ax.set_xlabel('Speaker')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Speaker Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Speaker {i}' for i in range(num_speakers)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_speaker_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Per-speaker comparison saved to: {output_dir / 'per_speaker_comparison.png'}")
    
    # Statistical significance test
    logger.info("\n" + "=" * 60)
    logger.info("Statistical Significance Test")
    logger.info("=" * 60)
    
    # Compute per-sample correctness
    rf_correct = (rf_results['predictions'] == y_test).astype(float)
    cnn_correct = (cnn_results['predictions'] == y_test_cnn).astype(float)
    
    stat_test = statistical_significance_test(
        rf_correct,
        cnn_correct,
        test=config.get('evaluation', {}).get('statistical_tests', {}).get('method', 'wilcoxon'),
        alpha=config.get('evaluation', {}).get('statistical_tests', {}).get('alpha', 0.05)
    )
    
    logger.info(f"\nTest: {stat_test['test']}")
    logger.info(f"Statistic: {stat_test['statistic']:.4f}")
    logger.info(f"P-value: {stat_test['p_value']:.4f}")
    logger.info(f"Significant: {stat_test['is_significant']}")
    logger.info(f"Mean difference: {stat_test['mean_difference']:.4f}")
    logger.info(f"\n{stat_test['interpretation']}")
    
    # Save all results
    results_summary = {
        'random_forest': rf_results['metrics'],
        'cnn': cnn_results['metrics'],
        'comparison': comparison_df.to_dict('records'),
        'statistical_test': stat_test
    }
    
    save_results(results_summary, output_dir / 'evaluation_summary.json')
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
