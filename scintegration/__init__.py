"""
scintegration: Integration Score Calculator for Single-Cell Batch Correction Evaluation

This package provides tools to evaluate batch correction methods in single-cell RNA-seq data
using an integration score that balances biology preservation against batch leakage.

Main Components:
- IntegrationScoreEvaluator: Main API for calculating integration scores
- ClusteringTask, MetadataLabelPredictionTask: Task implementations for benchmarking
- ClusteringTaskInput, MetadataLabelPredictionTaskInput: Task input classes
- calculate_integration_score: Core formula implementation
- normalize_silhouette: Normalization utility for silhouette scores
- plot_model_comparison, plot_metric_heatmap: Visualization functions
- configure_logging: Easy logging setup for debugging
- analyze_label_distribution: Check for rare classes before running benchmarks

Example Usage:
    >>> from scintegration import IntegrationScoreEvaluator, configure_logging, analyze_label_distribution
    >>> 
    >>> # Enable logging to see what's happening
    >>> configure_logging('INFO')
    >>> 
    >>> # Check for rare classes before running expensive benchmarks
    >>> analysis = analyze_label_distribution(
    ...     labels=adata.obs['cell_type'].values,
    ...     min_samples=5,
    ...     label_name="Cell Type"
    ... )
    >>> if analysis['has_rare_classes']:
    ...     print(f"{len(analysis['rare_classes'])} rare classes detected!")
    >>> 
    >>> # Simple API - evaluate embeddings directly
    >>> evaluator = IntegrationScoreEvaluator()
    >>> results = evaluator.evaluate_embeddings(
    ...     embeddings=scvi_embeddings,
    ...     obs=adata.obs,
    ...     biology_labels=cell_type_labels,
    ...     batch_labels=donor_labels,
    ...     model_name='scvi'
    ... )
    >>> print(f"Integration Score: {results.scores['scvi'].integration_score:.4f}")
    >>> print(f"Biology Score: {results.scores['scvi'].B:.4f}")
    >>> print(f"Leakage Score: {results.scores['scvi'].L:.4f}")
    >>> 
    >>> # Visualize results
    >>> from scintegration import plot_metric_heatmap
    >>> plot_metric_heatmap(results, model_names='scvi')

For more information, see the documentation at https://github.com/SkylarPurks/scintegration
"""

__version__ = "0.1.0"
__author__ = "SkylarPurks"

from .core import calculate_integration_score, compute_B_score, compute_L_score
from .normalization import normalize_silhouette, normalize_ari
from .evaluator import IntegrationScoreEvaluator, IntegrationScoreResults, ModelScore

# Import our own task implementations
from .tasks import (
    ClusteringTask,
    MetadataLabelPredictionTask,
    EmbeddingTask,
    ClusteringTaskInput,
    MetadataLabelPredictionTaskInput,
    EmbeddingTaskInput,
    MetricType,
    MetricResult,
)

# Import visualization functions
from .visualization import (
    plot_model_comparison,
    plot_metric_heatmap,
    plot_metric_summary,
)

# Import utility functions
from .utils import (
    interpret_integration_score,
    configure_logging,
    analyze_label_distribution,
    export_metrics_to_csv,
)

__all__ = [
    'IntegrationScoreEvaluator',
    'IntegrationScoreResults',
    'ModelScore',
    'calculate_integration_score',
    'compute_B_score',
    'compute_L_score',
    'normalize_silhouette',
    'normalize_ari',
    'ClusteringTask',
    'MetadataLabelPredictionTask',
    'ClusteringTaskInput',
    'MetadataLabelPredictionTaskInput',
    'EmbeddingTask',
    'EmbeddingTaskInput',
    'MetricType',
    'MetricResult',
    'plot_model_comparison',
    'plot_metric_heatmap',
    'plot_metric_summary',
    'configure_logging',
    'analyze_label_distribution',
    'interpret_integration_score',
    'export_metrics_to_csv',
]
