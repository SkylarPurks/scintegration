"""
Visualization functions for integration score results.

This module provides plotting functions to visualize and compare
integration scores across multiple models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
from .evaluator import IntegrationScoreResults, ModelScore


def plot_model_comparison(
    results: IntegrationScoreResults,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Create a bar chart comparing integration scores across multiple models.
    
    Parameters
    ----------
    results : IntegrationScoreResults
        Results object from IntegrationScoreEvaluator.evaluate()
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
    """
    # Extract data
    models = [score.model_name for score in results.get_ranked_models()]
    integration_scores = [score[1].integration_score for score in results.get_ranked_models()]
    B_scores = [score[1].B for score in results.get_ranked_models()]
    L_scores = [score[1].L for score in results.get_ranked_models()]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Color palette
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models)))
    
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


def plot_metric_heatmap(
    results: IntegrationScoreResults,
    model_names: Optional[Union[str, List[str]]] = None,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Create a heatmap showing biology vs batch metrics for each model.
    
    Parameters
    ----------
    results : IntegrationScoreResults
        Results object from IntegrationScoreEvaluator.evaluate()
    model_names : str or list of str, optional
        Model name(s) to plot. If None, plots all models
    metrics : list of str, optional
        Metrics to include. Default: ['ARI', 'NMI', 'Silhouette', 'F1']
    figsize : tuple, default=(8, 6)
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
    >>> from scintegration.visualization import plot_metric_heatmap
    >>> 
    >>> # Plot single model
    >>> fig = plot_metric_heatmap(results, model_names='scvi')
    >>> 
    >>> # Plot multiple models
    >>> fig = plot_metric_heatmap(results, model_names=['scvi', 'pca'])
    """
    # Default metrics
    if metrics is None:
        metrics = ['ARI', 'NMI', 'Silhouette', 'F1']
    
    # Handle model selection
    if model_names is None:
        model_names = list(results.scores.keys())
    elif isinstance(model_names, str):
        model_names = [model_names]
    
    # Single model heatmap
    if len(model_names) == 1:
        model_name = model_names[0]
        score = results.scores[model_name]
        
        # Extract metrics
        bio_values = []
        batch_values = []
        metric_labels = []
        
        metric_mapping = {
            'ARI': 'ari',
            'NMI': 'nmi',
            'Silhouette': 'silhouette',
            'F1': 'f1'
        }
        
        for metric in metrics:
            if metric in metric_mapping:
                key = metric_mapping[metric]
                if key in score.biology_components:
                    bio_values.append(score.biology_components[key])
                    batch_values.append(score.batch_components[key])
                    metric_labels.append(metric)
        
        # Create DataFrame
        df_heatmap = pd.DataFrame({
            'Biology (Preserve)': bio_values,
            'Batch (Remove)': batch_values
        }, index=metric_labels)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0, vmax=1, center=0.5, cbar_kws={'label': 'Score'},
                    linewidths=1, linecolor='white', ax=ax)
        
        ax.set_title(f'{model_name.upper()} Integration Quality: Metric Breakdown',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Evaluation Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        
    else:
        # Multiple model comparison heatmap
        # Create data for all models
        data_rows = []
        row_labels = []
        
        metric_mapping = {
            'ARI': 'ari',
            'NMI': 'nmi',
            'Silhouette': 'silhouette',
            'F1': 'f1'
        }
        
        for model_name in model_names:
            score = results.scores[model_name]
            for metric in metrics:
                if metric in metric_mapping:
                    key = metric_mapping[metric]
                    if key in score.biology_components and key in score.batch_components:
                        data_rows.append([
                            score.biology_components[key],
                            score.batch_components[key]
                        ])
                        row_labels.append(f"{model_name}_{metric}")
        
        # Create DataFrame
        df_heatmap = pd.DataFrame(
            data_rows,
            columns=['Biology (Preserve)', 'Batch (Remove)'],
            index=row_labels
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0, vmax=1, center=0.5, cbar_kws={'label': 'Score'},
                    linewidths=1, linecolor='white', ax=ax)
        
        ax.set_title('Multi-Model Integration Quality Comparison',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Evaluation Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model_Metric', fontsize=12, fontweight='bold')
    
    # Add interpretation text
    fig.text(0.5, 0.02,
             'High Biology scores (green) = good preservation. Low Batch scores (red) = good removal.',
             ha='center', fontsize=10, style='italic', wrap=True)
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        # Close figure from pyplot to prevent double display in notebooks
        plt.close(fig)
    
    return fig


def plot_metric_summary(
    results: IntegrationScoreResults,
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
    results : IntegrationScoreResults
        Results object from IntegrationScoreEvaluator.evaluate()
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
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
    
    # Left: Model comparison
    ax1 = fig.add_subplot(gs[0])
    ranked_models = results.get_ranked_models()
    models = [score[0] for score in ranked_models]
    integration_scores = [score[1].integration_score for score in ranked_models]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
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
