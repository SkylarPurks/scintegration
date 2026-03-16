"""
Metric normalization utilities.

This module provides functions to normalize various metrics to the [0, 1] range
where 1 represents the best possible value.
"""

import numpy as np
from typing import Union


def normalize_silhouette(sil_score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Normalize silhouette score from [-1, 1] to [0, 1].
    
    The silhouette coefficient ranges from -1 (worst) to +1 (best), where:
    - +1: Samples are well matched to their cluster and poorly matched to neighboring clusters
    - 0: Samples are on or very close to the decision boundary between clusters
    - -1: Samples may have been assigned to the wrong cluster
    
    This function applies a linear transformation to map this to [0, 1] range
    while preserving all information and maintaining the same relative ordering.
    
    Parameters
    ----------
    sil_score : float or np.ndarray
        Raw silhouette score(s) in range [-1, 1]
    
    Returns
    -------
    float or np.ndarray
        Normalized score(s) in range [0, 1], where 1 is best
    
    Examples
    --------
    >>> # Perfect clustering
    >>> normalize_silhouette(1.0)
    1.0
    
    >>> # Poor clustering
    >>> normalize_silhouette(-1.0)
    0.0
    
    >>> # Marginal clustering
    >>> normalize_silhouette(0.0)
    0.5
    
    >>> # Normalize multiple scores
    >>> import numpy as np
    >>> scores = np.array([0.8, 0.5, 0.0, -0.3])
    >>> normalize_silhouette(scores)
    array([0.9 , 0.75, 0.5 , 0.35])
    
    Notes
    -----
    This is a simple linear transformation: f(x) = (x + 1) / 2
    
    The transformation preserves:
    - Ordering: if sil_a > sil_b, then normalize(sil_a) > normalize(sil_b)
    - Intervals: the relative distance between scores is maintained
    - Information: the transformation is fully reversible
    
    Mathematical properties:
    - Domain: [-1, 1]
    - Range: [0, 1]
    - Monotonically increasing
    - Linear (preserves variance scaling)
    """
    return (sil_score + 1) / 2


def denormalize_silhouette(normalized_score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert normalized silhouette score from [0, 1] back to [-1, 1].
    
    This is the inverse operation of normalize_silhouette(), useful for
    interpreting results in the original silhouette scale.
    
    Parameters
    ----------
    normalized_score : float or np.ndarray
        Normalized silhouette score(s) in range [0, 1]
    
    Returns
    -------
    float or np.ndarray
        Original silhouette score(s) in range [-1, 1]
    
    Examples
    --------
    >>> # Convert back to original scale
    >>> denormalize_silhouette(1.0)
    1.0
    
    >>> denormalize_silhouette(0.5)
    0.0
    
    >>> denormalize_silhouette(0.35)
    -0.3
    """
    return 2 * normalized_score - 1


def normalize_ari(ari_score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Normalize Adjusted Rand Index from [-1, 1] to [0, 1].
    
    The ARI ranges from -1 (worst) to +1 (best), where:
    - +1: Perfect agreement between clusterings
    - 0: Random clustering (expected value for random labelings)
    - -1: Worse than random (systematically different clusterings)
    
    This function applies a linear transformation to map this to [0, 1] range.
    
    Parameters
    ----------
    ari_score : float or np.ndarray
        Raw ARI score(s) in range [-1, 1]
    
    Returns
    -------
    float or np.ndarray
        Normalized score(s) in range [0, 1], where 1 is best
    
    Examples
    --------
    >>> # Perfect clustering
    >>> normalize_ari(1.0)
    1.0
    
    >>> # Random clustering
    >>> normalize_ari(0.0)
    0.5
    
    >>> # Worse than random
    >>> normalize_ari(-0.5)
    0.25
    
    Notes
    -----
    This is the same linear transformation as silhouette: f(x) = (x + 1) / 2
    """
    return (ari_score + 1) / 2
    return 2 * normalized_score - 1
