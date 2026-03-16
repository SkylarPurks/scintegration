"""
Metric implementations for integration score evaluation.

This module provides metric calculation functions for evaluating biological
structure preservation and batch effect removal. These metrics are adapted from
cz-benchmarks to avoid external dependencies while maintaining compatibility.

References:
- https://github.com/chanzuckerberg/cz-benchmarks
"""

import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index between two clusterings.
    
    ARI measures the similarity between two clusterings, adjusted for chance.
    It ranges from -1 to 1, where 1 indicates perfect agreement, 0 indicates
    random labeling, and negative values indicate worse than random.
    
    Parameters
    ----------
    labels_true : np.ndarray
        Ground truth labels (e.g., cell types)
    labels_pred : np.ndarray
        Predicted cluster labels
        
    Returns
    -------
    float
        Adjusted Rand Index in range [-1, 1]
        
    References
    ----------
    Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of classification, 2(1), 193-218.
    """
    return adjusted_rand_score(labels_true, labels_pred)


def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information between two clusterings.
    
    NMI measures the mutual dependence between two clusterings, normalized
    to a [0, 1] scale where 1 indicates perfect agreement and 0 indicates
    no mutual information.
    
    Parameters
    ----------
    labels_true : np.ndarray
        Ground truth labels (e.g., cell types)
    labels_pred : np.ndarray
        Predicted cluster labels
        
    Returns
    -------
    float
        Normalized Mutual Information in range [0, 1]
        
    References
    ----------
    Strehl, A., & Ghosh, J. (2002). Cluster ensembles. Knowledge and information systems, 3(3), 305-344.
    """
    return normalized_mutual_info_score(labels_true, labels_pred)


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Silhouette score for clustering evaluation.
    
    The Silhouette score measures how well-separated clusters are. For each sample,
    it compares the mean distance to samples in its own cluster with the mean
    distance to samples in the nearest neighboring cluster.
    
    Parameters
    ----------
    X : np.ndarray
        Cell representation (embeddings or expression matrix), shape (n_cells, n_features)
    labels : np.ndarray
        Cluster labels for each cell
        
    Returns
    -------
    float
        Mean Silhouette score in range [-1, 1]
        - Values near 1 indicate well-separated clusters
        - Values near 0 indicate overlapping clusters
        - Negative values indicate misclassified samples
        
    Notes
    -----
    This implementation uses Euclidean distance by default. For integration
    score calculations, the result should be normalized to [0, 1] using:
    normalized_score = (score + 1) / 2
    
    References
    ----------
    Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation
    and validation of cluster analysis. Journal of computational and applied mathematics, 20, 53-65.
    """
    # Handle single cluster case
    if len(np.unique(labels)) < 2:
        return 0.0
    
    return silhouette_score(X, labels, metric='euclidean')


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Compute F1 score for classification evaluation.
    
    F1 is the harmonic mean of precision and recall. With macro averaging,
    it computes the metric independently for each class and takes the average,
    treating all classes equally.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    average : str, default='macro'
        Averaging strategy:
        - 'macro': Calculate metrics for each class and average (unweighted)
        - 'micro': Calculate metrics globally by counting total true positives, false negatives, etc.
        - 'weighted': Calculate metrics for each class and average, weighted by support
        - 'binary': Only for binary classification
        
    Returns
    -------
    float
        F1 score in range [0, 1], where 1 is perfect classification
        
    References
    ----------
    Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures
    for classification tasks. Information processing & management, 45(4), 427-437.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Accuracy is the proportion of correct predictions among total predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns
    -------
    float
        Accuracy in range [0, 1]
    """
    return accuracy_score(y_true, y_pred)


def compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Compute precision score.
    
    Precision is the ratio tp / (tp + fp) where tp is the number of true positives
    and fp the number of false positives.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    average : str, default='macro'
        Averaging strategy (see compute_f1 for details)
        
    Returns
    -------
    float
        Precision score in range [0, 1]
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Compute recall score.
    
    Recall is the ratio tp / (tp + fn) where tp is the number of true positives
    and fn the number of false negatives.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    average : str, default='macro'
        Averaging strategy (see compute_f1 for details)
        
    Returns
    -------
    float
        Recall score in range [0, 1]
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)
