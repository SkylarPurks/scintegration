"""
Utility functions for integration score analysis.

This module provides helper functions for visualization and interpretation
of integration scores.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def interpret_integration_score(score: float) -> str:
    """
    Provide a qualitative interpretation of an integration score.
    
    Parameters
    ----------
    score : float
        Integration score in range [-0.5, 0.5]
    
    Returns
    -------
    str
        Qualitative description of the score
    
    Examples
    --------
    >>> interpret_integration_score(0.25)
    'Excellent: Strong biology preservation with minimal batch leakage'
    
    >>> interpret_integration_score(0.05)
    'Good: Biology preservation exceeds batch leakage'
    
    >>> interpret_integration_score(-0.1)
    'Poor: Batch leakage exceeds biology preservation'
    """
    if score >= 0.2:
        return "Excellent: Strong biology preservation with minimal batch leakage"
    elif score >= 0.1:
        return "Very Good: Clear biology preservation over batch effects"
    elif score >= 0.05:
        return "Good: Biology preservation exceeds batch leakage"
    elif score >= 0.0:
        return "Marginal: Biology and batch effects are similar"
    elif score >= -0.1:
        return "Poor: Batch leakage exceeds biology preservation"
    else:
        return "Very Poor: Strong batch effects with weak biology signal"


def compute_theoretical_range(B_range: Tuple[float, float] = (0, 1),
                              L_range: Tuple[float, float] = (0, 1)) -> Tuple[float, float]:
    """
    Compute the theoretical min/max integration score for given B and L ranges.
    
    Parameters
    ----------
    B_range : tuple of float, default=(0, 1)
        Min and max possible values for biology score
    L_range : tuple of float, default=(0, 1)
        Min and max possible values for leakage score
    
    Returns
    -------
    tuple of float
        (min_IS, max_IS) theoretical integration score range
    
    Examples
    --------
    >>> compute_theoretical_range()
    (-0.5, 0.5)
    
    >>> # If we know biology is always reasonable
    >>> compute_theoretical_range(B_range=(0.5, 1.0))
    (0.0, 0.5)
    """
    B_min, B_max = B_range
    L_min, L_max = L_range
    
    # Maximum IS occurs at B=max, L=min
    if (B_max + L_min) > 0:
        max_IS = (B_max - L_min) / (2 * (B_max + L_min))
    else:
        max_IS = 0.0
    
    # Minimum IS occurs at B=min, L=max
    if (B_min + L_max) > 0:
        min_IS = (B_min - L_max) / (2 * (B_min + L_max))
    else:
        min_IS = 0.0
    
    return (min_IS, max_IS)


def batch_effect_percentage(L: float) -> float:
    """
    Convert leakage score to interpretable percentage of batch effect remaining.
    
    Parameters
    ----------
    L : float
        Batch leakage score [0, 1]
    
    Returns
    -------
    float
        Percentage of batch effect remaining (0-100)
    
    Examples
    --------
    >>> batch_effect_percentage(0.2)
    20.0
    
    >>> batch_effect_percentage(0.0)
    0.0
    """
    return L * 100


def biology_preservation_percentage(B: float) -> float:
    """
    Convert biology score to interpretable preservation percentage.
    
    Parameters
    ----------
    B : float
        Biology preservation score [0, 1]
    
    Returns
    -------
    float
        Percentage of biology preserved (0-100)
    
    Examples
    --------
    >>> biology_preservation_percentage(0.85)
    85.0
    
    >>> biology_preservation_percentage(0.7)
    70.0
    """
    return B * 100


def format_score_report(model_name: str, IS: float, B: float, L: float) -> str:
    """
    Format a comprehensive score report for a single model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    IS : float
        Integration score
    B : float
        Biology score
    L : float
        Leakage score
    
    Returns
    -------
    str
        Formatted report string
    
    Examples
    --------
    >>> print(format_score_report('scvi', 0.0702, 0.6970, 0.5597))
    MODEL: scvi
    ─────────────────────────────────────────
    Integration Score: +0.0702 (Good)
    Biology Preserved: 69.70%
    Batch Remaining:   55.97%
    Interpretation: Good integration
    """
    interpretation = interpret_integration_score(IS)
    bio_pct = biology_preservation_percentage(B)
    batch_pct = batch_effect_percentage(L)
    
    lines = [
        f"MODEL: {model_name}",
        "─" * 45,
        f"Integration Score: {IS:+.4f} ({interpretation.split(':')[0]})",
        f"Biology Preserved: {bio_pct:.2f}%",
        f"Batch Remaining:   {batch_pct:.2f}%",
        f"Interpretation: {interpretation}",
    ]
    
    return "\n".join(lines)


def configure_logging(level: str = 'INFO', format: Optional[str] = None) -> None:
    """
    Configure logging for the scintegration package.
    
    This is a convenience function to control how much detail you see in the logs.
    Call this at the start of your script to enable informative logging messages.
    
    Parameters
    ----------
    level : str, default='INFO'
        Logging level. Options:
        - 'DEBUG': Show all messages including detailed debugging info
        - 'INFO': Show general progress messages (recommended)
        - 'WARNING': Show only warnings and errors
        - 'ERROR': Show only errors
    format : str, optional
        Custom format string for log messages. If None, uses a sensible default.
        
    Examples
    --------
    >>> # Enable INFO-level logging (recommended for most users)
    >>> configure_logging('INFO')
    
    >>> # Enable DEBUG logging to see everything
    >>> configure_logging('DEBUG')
    
    >>> # Suppress most messages, only see warnings/errors
    >>> configure_logging('WARNING')
    
    >>> # Custom format
    >>> configure_logging('DEBUG', format='%(levelname)s - %(message)s')
    
    Notes
    -----
    This function configures logging for the entire 'scintegration' package.
    It's useful for debugging rare class issues, seeing which classes are filtered,
    and understanding what the tasks are doing internally.
    """
    if format is None:
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get the root logger for scintegration package
    logger = logging.getLogger('scintegration')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    logger.info(f"Logging configured for 'scintegration' package at {level.upper()} level")


def analyze_label_distribution(
    labels: np.ndarray,
    min_samples: int = 3,
    label_name: str = "Label"
) -> Dict[str, Any]:
    """
    Analyze the distribution of labels and identify potential issues for classification.
    
    This function helps you understand your data BEFORE running expensive benchmarks.
    It identifies rare classes that might cause cross-validation to fail and provides
    statistics about class balance.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of labels (cell types, batch labels, etc.)
    min_samples : int, default=3
        Minimum number of samples per class. Classes with fewer samples are flagged
        as "rare" and may cause issues during cross-validation (especially with n_folds > 3).
    label_name : str, default="Label"
        Descriptive name for the labels (e.g., "Cell Type", "Batch", "Donor")
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'n_classes': Total number of unique classes
        - 'n_samples': Total number of samples
        - 'class_counts': Pandas Series with counts for each class (sorted)
        - 'rare_classes': List of classes with < min_samples samples
        - 'has_rare_classes': Boolean indicating if rare classes exist
        - 'min_class_size': Size of the smallest class
        - 'max_class_size': Size of the largest class
        - 'imbalance_ratio': max_class_size / min_class_size
        
    Examples
    --------
    >>> # Check your cell type labels before running benchmarks
    >>> from scintegration.utils import analyze_label_distribution
    >>> 
    >>> analysis = analyze_label_distribution(
    ...     labels=adata.obs['cell_type'].values,
    ...     min_samples=5,  # Flag classes with < 5 samples
    ...     label_name="Cell Type"
    ... )
    >>> 
    >>> # Print the summary
    >>> print(f"Total classes: {analysis['n_classes']}")
    >>> print(f"Rare classes: {len(analysis['rare_classes'])}")
    >>> if analysis['has_rare_classes']:
    ...     print(f"⚠️ Warning: {len(analysis['rare_classes'])} rare classes detected:")
    ...     for cls in analysis['rare_classes']:
    ...         count = analysis['class_counts'][cls]
    ...         print(f"  - {cls}: {count} samples (need {min_samples})")
    
    >>> # Use it to filter your data
    >>> if analysis['has_rare_classes']:
    ...     # Remove rare classes before running benchmarks
    ...     valid_mask = np.isin(adata.obs['cell_type'], analysis['rare_classes'], invert=True)
    ...     adata_filtered = adata[valid_mask].copy()
    
    Notes
    -----
    Use this function to:
    1. Check for rare classes before running MetadataLabelPredictionTask
    2. Determine appropriate min_class_size parameter
    3. Decide if you need to filter or merge rare classes
    4. Understand class imbalance in your dataset
    
    For cross-validation with n_folds=5, you need at least 5 samples per class.
    For n_folds=3, you need at least 3 samples per class.
    """
    # Convert to numpy array if needed
    labels = np.asarray(labels)
    
    # Get unique labels and counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Create a sorted pandas Series for easy inspection
    class_counts = pd.Series(counts, index=unique_labels).sort_values(ascending=False)
    
    # Identify rare classes
    rare_classes = unique_labels[counts < min_samples].tolist()
    
    # Calculate statistics
    n_classes = len(unique_labels)
    n_samples = len(labels)
    min_class_size = int(np.min(counts))
    max_class_size = int(np.max(counts))
    imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
    
    # Create summary dictionary
    analysis = {
        'n_classes': n_classes,
        'n_samples': n_samples,
        'class_counts': class_counts,
        'rare_classes': rare_classes,
        'has_rare_classes': len(rare_classes) > 0,
        'min_class_size': min_class_size,
        'max_class_size': max_class_size,
        'imbalance_ratio': imbalance_ratio,
    }
    
    # Log summary
    logger = logging.getLogger('scintegration.utils')
    logger.info(f"\n{label_name} Distribution Analysis:")
    logger.info(f"  Total {label_name}s: {n_classes}")
    logger.info(f"  Total samples: {n_samples}")
    logger.info(f"  Class size range: [{min_class_size}, {max_class_size}]")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if len(rare_classes) > 0:
        logger.warning(f"  ⚠️  {len(rare_classes)} rare classes (< {min_samples} samples):")
        for cls in rare_classes[:10]:  # Show first 10
            logger.warning(f"    - {cls}: {class_counts[cls]} samples")
        if len(rare_classes) > 10:
            logger.warning(f"    ... and {len(rare_classes) - 10} more")
    else:
        logger.info(f"  ✓ No rare classes detected (all have ≥ {min_samples} samples)")
    
    return analysis


def export_metrics_to_csv(
    data: pd.DataFrame,
    model_name: Optional[str] = None,
    output_dir: str = ".",
    filename: Optional[str] = None
) -> str:
    """
    Export integration metrics to a CSV file with automatic naming based on model and date.
    
    This function saves evaluation metrics to a CSV file with a standardized filename that
    includes the model name (if provided) and the current date and time. This makes it easy
    to track and compare results from different runs.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the metrics to export. Can be obtained from
        IntegrationScoreResults.to_dataframe() or any custom DataFrame
    model_name : str, optional
        Name of the model being evaluated. If provided, will be included in the filename.
        If None, uses "all_models" for multi-model results.
    output_dir : str, default="."
        Directory where the CSV file will be saved. Will be created if it doesn't exist.
    filename : str, optional
        Custom filename for the CSV file. If provided, overrides automatic naming.
        model_name and date will not be appended if filename is specified.
    
    Returns
    -------
    str
        Path to the saved CSV file
    
    Examples
    --------
    >>> from scintegration import IntegrationScoreEvaluator
    >>> from scintegration.utils import export_metrics_to_csv
    >>> 
    >>> # Evaluate models
    >>> evaluator = IntegrationScoreEvaluator()
    >>> results = evaluator.evaluate(...)
    >>> 
    >>> # Export all models to CSV with automatic naming
    >>> df = results.to_dataframe()
    >>> csv_path = export_metrics_to_csv(df, output_dir="results")
    >>> print(f"Metrics saved to: {csv_path}")
    >>> # Output: results/integration_metrics_all_models_2026-03-09.csv
    
    >>> # Export single model results
    >>> scvi_data = df[df['model_name'] == 'scvi']
    >>> csv_path = export_metrics_to_csv(scvi_data, model_name='scvi', output_dir="results/scvi")
    >>> # Output: results/scvi/integration_metrics_scvi_2026-03-09.csv
    
    >>> # Use custom filename
    >>> csv_path = export_metrics_to_csv(df, filename="my_results.csv")
    >>> # Output: my_results.csv
    
    Notes
    -----
    - The default filename format is: integration_metrics_{model_name}_{date}.csv
    - Date format is: YYYY-MM-DD
    - The output directory will be created if it doesn't exist
    - Existing files with the same name will be overwritten
    
    See Also
    --------
    IntegrationScoreResults.to_csv : Convenience method that uses this function
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        # Get current date
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Determine model name for filename
        if model_name is None:
            model_part = "all_models"
        else:
            # Sanitize model name for filename (replace spaces and special chars)
            model_part = model_name.replace(" ", "_").replace("/", "-")
        
        filename = f"integration_metrics_{model_part}_{date_str}.csv"
    
    # Construct full path
    full_path = output_path / filename
    
    # Save to CSV
    data.to_csv(full_path, index=False)
    
    logger = logging.getLogger('scintegration.utils')
    logger.info(f"Metrics exported to: {full_path}")
    
    return str(full_path)

