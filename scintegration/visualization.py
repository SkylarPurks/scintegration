"""
Visualization functions for integration score results.

This module provides plotting functions to visualize and compare
integration scores across multiple models. Functions accept either
CSV file paths or IntegrationScoreResults objects as input.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Optional, List, Union
from .evaluator import IntegrationScoreResults, ModelScore
from .normalization import denormalize_silhouette


# Helper functions for CSV loading
def _safe_float(value):
    """Convert value to float, handling NaN and string representations."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def _load_and_clean_results_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and clean a results CSV file.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with numeric columns converted to float
    """
    csv_path = Path(csv_path)
    numeric_columns = [
        "integration_score", "B", "L",
        "bio_ari", "bio_nmi", "bio_silhouette", "bio_f1",
        "batch_ari", "batch_nmi", "batch_silhouette", "batch_f1"
    ]
    
    df = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip")
    df.columns = [str(col).strip() for col in df.columns]
    
    # Convert numeric columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(_safe_float)
    
    # Validate required columns
    required_columns = [
        "model_name", "integration_score", "B", "L",
        "bio_ari", "bio_nmi", "bio_silhouette", "bio_f1",
        "batch_ari", "batch_nmi", "batch_silhouette"
    ]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_required}")
    
    # Skip malformed rows
    invalid_mask = df[required_columns].isna().any(axis=1)
    if invalid_mask.any():
        print(f"Skipping {invalid_mask.sum()} malformed row(s) in {csv_path.name}")
        df = df.loc[~invalid_mask].copy()
    
    if df.empty:
        raise ValueError(f"No valid rows remained after cleaning {csv_path}")
    
    return df.sort_values("integration_score", ascending=False).reset_index(drop=True)


def _integration_results_from_csv(csv_path: Union[str, Path]) -> IntegrationScoreResults:
    """
    Load a CSV file and convert to IntegrationScoreResults object.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file
    
    Returns
    -------
    IntegrationScoreResults
        Results object ready for visualization
    """
    df = _load_and_clean_results_csv(csv_path)
    scores = {}
    
    for _, row in df.iterrows():
        biology_components = {
            "ari": _safe_float(row["bio_ari"]),
            "nmi": _safe_float(row["bio_nmi"]),
            "silhouette": _safe_float(row["bio_silhouette"]),
            "f1": _safe_float(row["bio_f1"])
        }
        batch_components = {
            "ari": _safe_float(row["batch_ari"]),
            "nmi": _safe_float(row["batch_nmi"]),
            "silhouette": _safe_float(row["batch_silhouette"])
        }
        batch_f1 = _safe_float(row.get("batch_f1"))
        if not pd.isna(batch_f1):
            batch_components["f1"] = batch_f1
        
        scores[row["model_name"]] = ModelScore(
            model_name=row["model_name"],
            integration_score=_safe_float(row["integration_score"]),
            B=_safe_float(row["B"]),
            L=_safe_float(row["L"]),
            biology_components=biology_components,
            batch_components=batch_components
        )
    
    return IntegrationScoreResults(scores)


def _ensure_results_object(
    results: Union[str, Path, IntegrationScoreResults]
) -> IntegrationScoreResults:
    """
    Convert input to IntegrationScoreResults object if needed.
    
    Parameters
    ----------
    results : str, Path, or IntegrationScoreResults
        Either a CSV file path or a results object
    
    Returns
    -------
    IntegrationScoreResults
        Results object ready for visualization
    """
    if isinstance(results, IntegrationScoreResults):
        return results
    else:
        return _integration_results_from_csv(results)


def _get_model_colors(model_names: List[str]) -> Dict[str, tuple]:
    """
    Generate a consistent color mapping for models.
    
    Uses a deterministic color scheme based on model names to ensure
    the same model always gets the same color across different plots.
    
    Parameters
    ----------
    model_names : list of str
        List of model names
    
    Returns
    -------
    dict
        Dictionary mapping model name to RGB color tuple
    """
    # Sort model names to ensure consistent ordering
    sorted_names = sorted(model_names)
    
    # Generate colors from RdYlGn colormap
    colors_array = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_names)))
    
    # Map each model name to its color
    color_map = {name: colors_array[i] for i, name in enumerate(sorted_names)}
    
    return color_map


def plot_model_comparison(
    results: Union[str, Path, IntegrationScoreResults],
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Create a bar chart comparing integration scores across multiple models.
    
    Parameters
    ----------
    results : str, Path, or IntegrationScoreResults
        Either a CSV file path or a results object from IntegrationScoreEvaluator.evaluate()
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    show : bool, default=False
        Whether to call plt.show(). In Jupyter notebooks, returned figures
        are automatically displayed, so this should remain False
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> from scintegration import IntegrationScoreEvaluator
    >>> from scintegration.visualization import plot_model_comparison
    >>> 
    >>> evaluator = IntegrationScoreEvaluator()
    >>> results = evaluator.evaluate(...)
    >>> fig = plot_model_comparison(results)
    >>> 
    >>> # Or use a CSV file
    >>> fig = plot_model_comparison('results.csv')
    """
    results = _ensure_results_object(results)
    # Extract data
    ranked_models = results.get_ranked_models()
    models = [score[0] for score in ranked_models]
    integration_scores = [score[1].integration_score for score in ranked_models]
    B_scores = [score[1].B for score in ranked_models]
    L_scores = [score[1].L for score in ranked_models]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get consistent colors for models
    model_color_map = _get_model_colors(list(results.scores.keys()))
    colors = [model_color_map[model] for model in models]
    
    # Plot 1: Integration Score
    axes[0].barh(models, integration_scores, color=colors)
    axes[0].set_xlabel('Integration Score', fontweight='bold')
    axes[0].set_title('Integration Score\n(Higher is Better)', fontsize=11, fontweight='bold')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0].set_xlim(-0.5, 0.5)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Biology Score (B)
    axes[1].barh(models, B_scores, color='green', alpha=0.6)
    axes[1].set_xlabel('Biology Score (B)', fontweight='bold')
    axes[1].set_title('Biology Preservation\n(Higher is Better)', fontsize=11, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Leakage Score (L)
    axes[2].barh(models, L_scores, color='red', alpha=0.6)
    axes[2].set_xlabel('Leakage Score (L)', fontweight='bold')
    axes[2].set_title('Batch Leakage\n(Lower is Better)', fontsize=11, fontweight='bold')
    axes[2].set_xlim(0, 1)
    axes[2].grid(axis='x', alpha=0.3)
    
    # Overall title
    fig.suptitle('Model Comparison: Integration Quality', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        # Close figure from pyplot to prevent double display in notebooks
        plt.close(fig)
    
    return fig


def plot_metric_bar_comparison(
    results: Union[str, Path, IntegrationScoreResults],
    model_names: Optional[Union[str, List[str]]] = None,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    split_legend: bool = True,
    batch_hatch: str = '///',
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Create a grouped bar chart comparing biology and batch metrics across models.

    This plot compares biology and batch metrics with grouped bars.
    Silhouette values are displayed in their original range [-1, 1].

    Parameters
    ----------
    results : str, Path, or IntegrationScoreResults
        Either a CSV file path or a results object from IntegrationScoreEvaluator.evaluate()
    model_names : str or list of str, optional
        Model name(s) to plot. If None, plots all models
    metrics : list of str, optional
        Metrics to include. Default: ['ARI', 'NMI', 'Silhouette', 'F1']
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
    split_legend : bool, default=True
        If True, use separate legends for model colors and Biology/Batch style.
        If False, use the older combined legend with one entry per series.
    batch_hatch : str, default='///'
        Hatch pattern used for batch bars when split_legend is enabled.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    show : bool, default=False
        Whether to call plt.show(). In Jupyter notebooks, returned figures
        are automatically displayed, so this should remain False

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    results = _ensure_results_object(results)
    if metrics is None:
        metrics = ['ARI', 'NMI', 'Silhouette', 'F1']

    if model_names is None:
        model_names = list(results.scores.keys())
    elif isinstance(model_names, str):
        model_names = [model_names]

    metric_mapping = {
        'ARI': 'ari',
        'NMI': 'nmi',
        'Silhouette': 'silhouette',
        'F1': 'f1'
    }

    valid_metrics = [m for m in metrics if m in metric_mapping]
    if not valid_metrics:
        raise ValueError("No valid metrics selected. Use metrics from: ARI, NMI, Silhouette, F1")

    n_metrics = len(valid_metrics)
    x = np.arange(n_metrics)
    n_series = len(model_names) * 2  # Biology + Batch for each model
    total_width = 0.8
    bar_width = total_width / max(n_series, 1)

    fig, ax = plt.subplots(figsize=figsize)
    model_color_map = _get_model_colors(model_names)

    series_index = 0
    for model_idx, model_name in enumerate(model_names):
        score = results.scores[model_name]

        bio_vals = []
        batch_vals = []
        for metric in valid_metrics:
            key = metric_mapping[metric]

            bio_val = score.biology_components.get(key, np.nan)
            batch_val = score.batch_components.get(key, np.nan)

            if metric == 'Silhouette':
                if pd.notna(bio_val):
                    bio_val = denormalize_silhouette(bio_val)
                if pd.notna(batch_val):
                    batch_val = denormalize_silhouette(batch_val)

            bio_vals.append(bio_val)
            batch_vals.append(batch_val)

        bio_offset = -total_width / 2 + (series_index + 0.5) * bar_width
        ax.bar(
            x + bio_offset,
            bio_vals,
            width=bar_width,
            label=f'{model_name} Biology',
            color=model_color_map[model_name],
            alpha=0.9
        )
        series_index += 1

        batch_offset = -total_width / 2 + (series_index + 0.5) * bar_width
        ax.bar(
            x + batch_offset,
            batch_vals,
            width=bar_width,
            label=f'{model_name} Batch' if not split_legend else None,
            color=model_color_map[model_name],
            alpha=0.45,
            hatch=batch_hatch if split_legend else None,
            edgecolor='black' if split_legend else None,
            linewidth=0.2 if split_legend else 0
        )
        series_index += 1

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_metrics)
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Model Metric Comparison (Bar Chart)',
                 fontsize=14, fontweight='bold', pad=18)
    ax.grid(axis='y', alpha=0.25)

    if split_legend:
        # Legend 1: model identity by color
        model_handles = [
            Line2D([0], [0], color=model_color_map[m], lw=6, label=m)
            for m in model_names
        ]
        leg1 = ax.legend(
            handles=model_handles,
            title='Model (color)',
            loc='upper left',
            bbox_to_anchor=(1.01, 1),
            frameon=True
        )
        ax.add_artist(leg1)

        # Legend 2: component identity by style
        style_handles = [
            Patch(facecolor='gray', alpha=0.9, label='Biology'),
            Patch(facecolor='gray', alpha=0.45, hatch=batch_hatch, edgecolor='black', label='Batch'),
        ]
        ax.legend(
            handles=style_handles,
            title='Component (style)',
            loc='upper left',
            bbox_to_anchor=(1.01, 0.45),
            frameon=True
        )
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)

    fig.tight_layout(rect=[0, 0.06, 0.84, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_metric_summary(
    results: Union[str, Path, IntegrationScoreResults],
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Create a summary visualization with integration scores and metric breakdown.
    
    Combines a bar chart of integration scores with a detailed metric breakdown
    for the best performing model.
    
    Parameters
    ----------
    results : str, Path, or IntegrationScoreResults
        Either a CSV file path or a results object from IntegrationScoreEvaluator.evaluate()
    figsize : tuple, default=(12, 5)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    show : bool, default=False
        Whether to call plt.show(). In Jupyter notebooks, returned figures
        are automatically displayed, so this should remain False
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    results = _ensure_results_object(results)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
    
    # Left: Model comparison
    ax1 = fig.add_subplot(gs[0])
    ranked_models = results.get_ranked_models()
    models = [score[0] for score in ranked_models]
    integration_scores = [score[1].integration_score for score in ranked_models]
    
    # Get consistent colors for models
    model_color_map = _get_model_colors(list(results.scores.keys()))
    colors = [model_color_map[model] for model in models]
    bars = ax1.barh(models, integration_scores, color=colors)
    
    # Highlight best model
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)
    
    ax1.set_xlabel('Integration Score', fontweight='bold', fontsize=11)
    ax1.set_title('Model Ranking', fontsize=12, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (model, score_val) in enumerate(zip(models, integration_scores)):
        ax1.text(score_val, i, f'  {score_val:.3f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # Right: Best model metric breakdown
    ax2 = fig.add_subplot(gs[1])
    best_model_name, best_score = ranked_models[0]
    
    all_metric_labels = ['ARI', 'NMI', 'Sil', 'F1']
    all_metric_keys = ['ari', 'nmi', 'silhouette', 'f1']
    # Only include metrics present in both components (F1 absent for single-batch)
    metrics = []
    metric_keys = []
    for label, key in zip(all_metric_labels, all_metric_keys):
        if key in best_score.biology_components and key in best_score.batch_components:
            metrics.append(label)
            metric_keys.append(key)
    
    bio_vals = [best_score.biology_components[k] for k in metric_keys]
    batch_vals = [best_score.batch_components[k] for k in metric_keys]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, bio_vals, width, label='Biology (High=Good)', 
            color='green', alpha=0.7)
    ax2.bar(x + width/2, batch_vals, width, label='Batch (Low=Good)', 
            color='red', alpha=0.7)
    
    ax2.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax2.set_title(f'Best Model: {best_model_name.upper()}\n(IS={best_score.integration_score:.3f})',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Integration Score Summary', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        # Close figure from pyplot to prevent double display in notebooks
        plt.close(fig)
    
    return fig
