"""
Core integration score calculation logic.

This module contains the fundamental formulas for computing integration scores
from biology preservation and batch leakage scores.
"""

import numpy as np
from typing import Union, Dict, Optional


def calculate_integration_score(B: float, L: float) -> float:
    """
    Calculate integration score from biology and leakage scores.
    
    The integration score quantifies how well a batch correction method preserves
    biological signal while removing batch effects. It ranges from -0.5 to +0.5,
    where higher values indicate better integration.
    
    Formula: IS = (B - L) / (2 * (B + L))
    
    Parameters
    ----------
    B : float
        Biology preservation score [0, 1], where 1 indicates perfect preservation
        of biological structure (cell types, states, etc.)
    L : float
        Batch leakage score [0, 1], where 0 indicates no batch signal remaining
        and 1 indicates complete batch separation
    
    Returns
    -------
    float
        Integration score in range [-0.5, 0.5]
        - IS > 0.2: Excellent integration (biology >> batch)
        - IS > 0.0: Good integration (biology > batch)
        - IS ≈ 0.0: Marginal (biology ≈ batch)
        - IS < 0.0: Poor (batch > biology)
    
    Examples
    --------
    >>> # Perfect integration: high biology, low batch
    >>> calculate_integration_score(B=0.8, L=0.2)
    0.3
    
    >>> # Poor integration: low biology, high batch
    >>> calculate_integration_score(B=0.3, L=0.7)
    -0.2
    
    >>> # Marginal: biology and batch equal
    >>> calculate_integration_score(B=0.5, L=0.5)
    0.0
    
    Notes
    -----
    If B + L = 0 (both scores are zero), returns 0.0 to avoid division by zero.
    This edge case should rarely occur in practice.
    """
    if (B + L) == 0:
        return 0.0
    return (B - L) / (2 * (B + L))


def compute_B_score(
    ari_bio: float,
    nmi_bio: float,
    sil_bio: float,
    f1_bio: Optional[float] = None,
    weights: Union[str, Dict[str, float]] = 'equal'
) -> float:
    """
    Compute biology preservation score from individual metrics.
    
    The B score quantifies how well biological structure (e.g., cell types,
    cell states) is preserved after batch correction. It combines clustering
    agreement metrics (ARI, NMI, Silhouette) with classification performance (F1).
    
    Parameters
    ----------
    ari_bio : float
        Adjusted Rand Index for biological labels [0, 1]
        Measures clustering agreement with known cell types
    nmi_bio : float
        Normalized Mutual Information for biological labels [0, 1]
        Measures information shared between clusters and cell types
    sil_bio : float
        Normalized silhouette score for biological labels [0, 1]
        Measures cluster cohesion and separation (must be pre-normalized from [-1,1])
    f1_bio : float, optional
        F1 score for biological label classification [0, 1]
        Macro-averaged F1 across all cell types. If None, only clustering metrics are used.
    weights : str or dict, default='equal'
        Weighting scheme for combining metrics:
        - 'equal': Simple average of all metrics (recommended)
        - dict: Custom weights, e.g., {'ari': 0.3, 'nmi': 0.3, 'sil': 0.2, 'f1': 0.2}
    
    Returns
    -------
    float
        Biology preservation score [0, 1], where 1 is perfect preservation
    
    Examples
    --------
    >>> # High biology preservation
    >>> compute_B_score(ari_bio=0.85, nmi_bio=0.82, sil_bio=0.78, f1_bio=0.80)
    0.8125
    
    >>> # Custom weights emphasizing clustering metrics
    >>> weights = {'ari': 0.3, 'nmi': 0.3, 'sil': 0.3, 'f1': 0.1}
    >>> compute_B_score(0.85, 0.82, 0.78, 0.80, weights=weights)
    0.815
    
    Notes
    -----
    All input metrics should be normalized to [0, 1] range before calling this function.
    For silhouette scores, use `normalize_silhouette()` to convert from [-1, 1] to [0, 1].
    """
    if weights == 'equal':
        if f1_bio is None:
            return np.mean([ari_bio, nmi_bio, sil_bio])
        return np.mean([ari_bio, nmi_bio, sil_bio, f1_bio])
    else:
        # Custom weights
        if f1_bio is None:
            # Exclude F1 from calculation
            total = (
                weights.get('ari', 0.25) * ari_bio +
                weights.get('nmi', 0.25) * nmi_bio +
                weights.get('sil', 0.25) * sil_bio
            )
            # Renormalize if weights were specified
            weight_sum = weights.get('ari', 0.25) + weights.get('nmi', 0.25) + weights.get('sil', 0.25)
            if weight_sum > 0:
                total = total / weight_sum
        else:
            total = (
                weights.get('ari', 0.25) * ari_bio +
                weights.get('nmi', 0.25) * nmi_bio +
                weights.get('sil', 0.25) * sil_bio +
                weights.get('f1', 0.25) * f1_bio
            )
        return total


def compute_L_score(
    ari_batch: float,
    nmi_batch: float,
    sil_batch: float,
    f1_batch: Optional[float] = None,
    weights: Union[str, Dict[str, float]] = 'equal'
) -> float:
    """
    Compute batch leakage score from individual metrics.
    
    The L score quantifies how much batch information remains after correction.
    Lower scores are better (indicating less batch effect). It combines clustering
    agreement metrics (ARI, NMI, Silhouette) with classification performance (F1)
    on batch labels (donors, sequencing batches, etc.).
    
    Parameters
    ----------
    ari_batch : float
        Adjusted Rand Index for batch labels [0, 1]
        Measures clustering agreement with batch identities
    nmi_batch : float
        Normalized Mutual Information for batch labels [0, 1]
        Measures information shared between clusters and batches
    sil_batch : float
        Normalized silhouette score for batch labels [0, 1]
        Measures cluster cohesion and separation by batch
    f1_batch : float, optional
        F1 score for batch label classification [0, 1]
        Macro-averaged F1 across all batches. If None (e.g., single batch), only clustering metrics are used.
    weights : str or dict, default='equal'
        Weighting scheme for combining metrics (same as compute_B_score)
    
    Returns
    -------
    float
        Batch leakage score [0, 1], where 0 is perfect batch removal
    
    Examples
    --------
    >>> # Good batch removal (low leakage)
    >>> compute_L_score(ari_batch=0.15, nmi_batch=0.12, sil_batch=0.18, f1_batch=0.20)
    0.1625
    
    >>> # Poor batch removal (high leakage)
    >>> compute_L_score(ari_batch=0.65, nmi_batch=0.70, sil_batch=0.68, f1_batch=0.75)
    0.695
    
    >>> # Single batch case (F1 excluded)
    >>> compute_L_score(ari_batch=0.15, nmi_batch=0.12, sil_batch=0.18, f1_batch=None)
    0.15
    
    Notes
    -----
    This function uses the same logic as compute_B_score but interprets the result
    differently: high L values indicate poor batch correction (batch signal remains).
    """
    if weights == 'equal':
        if f1_batch is None:
            return np.mean([ari_batch, nmi_batch, sil_batch])
        return np.mean([ari_batch, nmi_batch, sil_batch, f1_batch])
    else:
        # Custom weights
        if f1_batch is None:
            # Exclude F1 from calculation
            total = (
                weights.get('ari', 0.25) * ari_batch +
                weights.get('nmi', 0.25) * nmi_batch +
                weights.get('sil', 0.25) * sil_batch
            )
            # Renormalize if weights were specified
            weight_sum = weights.get('ari', 0.25) + weights.get('nmi', 0.25) + weights.get('sil', 0.25)
            if weight_sum > 0:
                total = total / weight_sum
        else:
            total = (
                weights.get('ari', 0.25) * ari_batch +
                weights.get('nmi', 0.25) * nmi_batch +
                weights.get('sil', 0.25) * sil_batch +
                weights.get('f1', 0.25) * f1_batch
            )
        return total
