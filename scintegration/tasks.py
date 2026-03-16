"""
Task implementations for integration score evaluation.

This module provides task classes that wrap metric calculations,
implementing the same API as czbenchmarks for compatibility with existing code.

References:
- https://github.com/chanzuckerberg/cz-benchmarks
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import anndata as ad
import scanpy as sc
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

# Setup logger
logger = logging.getLogger(__name__)

from .metrics import (
    compute_ari,
    compute_nmi,
    compute_silhouette,
    compute_f1,
    compute_accuracy,
    compute_precision,
    compute_recall,
)


class MetricType(Enum):
    """Enumeration of supported metric types."""
    ADJUSTED_RAND_INDEX = "adjusted_rand_index"
    NORMALIZED_MUTUAL_INFO = "normalized_mutual_info"
    SILHOUETTE_SCORE = "silhouette_score"
    MEAN_FOLD_F1_SCORE = "mean_fold_f1"
    MEAN_FOLD_ACCURACY = "mean_fold_accuracy"
    MEAN_FOLD_PRECISION = "mean_fold_precision"
    MEAN_FOLD_RECALL = "mean_fold_recall"
    MEAN_FOLD_AUROC = "mean_fold_auroc"


@dataclass
class MetricResult:
    """Container for a single metric result."""
    metric_type: MetricType
    value: float
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class ClusteringTaskInput:
    """Input for clustering task."""
    obs: Optional[pd.DataFrame] = None
    input_labels: Optional[np.ndarray] = None


@dataclass
class EmbeddingTaskInput:
    """Input for embedding task."""
    input_labels: np.ndarray


@dataclass
class MetadataLabelPredictionTaskInput:
    """
    Input for label prediction task.
    
    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels for each cell
    n_folds : int, default=5
        Number of folds for stratified cross-validation
    min_class_size : int, default=10
        Minimum number of samples required per class for inclusion in evaluation
    
    Notes
    -----
    If you encounter issues with too few classes being used:
    - Check your label distribution with `pd.Series(labels).value_counts()`
    - Adjust `min_class_size` parameter if you have small classes you want to include
    - Consider using a more balanced dataset if classes are heavily imbalanced
    
    The task will return an empty list if fewer than 2 classes remain after filtering.
    """
    labels: np.ndarray
    n_folds: int = 5
    min_class_size: int = 10
    
    def __post_init__(self):
        """Validate input parameters."""
        if self.n_folds <= 0:
            raise ValueError(f"n_folds must be a positive integer, got {self.n_folds}")
        
        if self.min_class_size <= 0:
            raise ValueError(f"min_class_size must be a positive integer, got {self.min_class_size}")
        
        if len(self.labels) == 0:
            raise ValueError("labels must not be empty")
        
        if self.n_folds > len(self.labels):
            raise ValueError(
                f"n_folds ({self.n_folds}) cannot exceed number of samples ({len(self.labels)})"
            )
        
        logger.debug(f"MetadataLabelPredictionTaskInput initialized: n_samples={len(self.labels)}, "
                     f"n_folds={self.n_folds}, min_class_size={self.min_class_size}")


class ClusteringTask:
    """
    Task for evaluating clustering performance.
    
    Performs Leiden clustering on embeddings and evaluates results using
    Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
    
    This implementation follows the czbenchmarks API for compatibility.
    
    Parameters
    ----------
    random_seed : int, default=42
        Random seed for reproducibility
    n_neighbors : int, default=15
        Number of neighbors for k-NN graph construction.
        Higher values create more connected graphs (smoother clusters).
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering.
        Higher values yield more fine-grained clusters.
    """
    
    def __init__(self, random_seed: int = 42, n_neighbors: int = 15, resolution: float = 1.0):
        self.random_seed = random_seed
        self.n_neighbors = n_neighbors
        self.resolution = resolution
    
    def run(
        self,
        cell_representation: np.ndarray,
        task_input: ClusteringTaskInput
    ) -> List[MetricResult]:
        """
        Run clustering task and compute metrics.
        
        Parameters
        ----------
        cell_representation : np.ndarray
            Cell embeddings, shape (n_cells, n_features)
        task_input : ClusteringTaskInput
            Task input containing ground truth labels
            
        Returns
        -------
        List[MetricResult]
            List of metric results (ARI and NMI)
        """
        logger.info(f"Starting ClusteringTask with {cell_representation.shape[0]} cells, "
                   f"{cell_representation.shape[1]} features")
        
        # Create AnnData object for scanpy's Leiden clustering
        adata = ad.AnnData(X=cell_representation)
        
        # Compute neighbors and run Leiden clustering
        logger.debug(f"Computing neighbors with n_neighbors={self.n_neighbors}...")
        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, random_state=self.random_seed)
        
        logger.debug(f"Running Leiden clustering with resolution={self.resolution}...")
        sc.tl.leiden(adata, resolution=self.resolution, random_state=self.random_seed)
        
        # Get predicted cluster labels
        predicted_labels = adata.obs['leiden'].to_numpy()
        logger.info(f"Leiden clustering identified {len(np.unique(predicted_labels))} clusters")
        
        # Compute metrics
        true_labels = task_input.input_labels
        ari = compute_ari(true_labels, predicted_labels)
        nmi = compute_nmi(true_labels, predicted_labels)
        
        logger.info(f"Clustering metrics: ARI={ari:.4f}, NMI={nmi:.4f}")
        
        results = [
            MetricResult(
                metric_type=MetricType.ADJUSTED_RAND_INDEX,
                value=ari
            ),
            MetricResult(
                metric_type=MetricType.NORMALIZED_MUTUAL_INFO,
                value=nmi
            )
        ]
        
        return results


class EmbeddingTask:
    """
    Task for evaluating embedding quality.
    
    Computes the Silhouette score to measure how well-separated clusters
    are in the embedding space using ground truth labels.
    
    Parameters
    ----------
    random_seed : int, default=42
        Random seed for reproducibility (unused but kept for API compatibility)
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def run(
        self,
        cell_representation: np.ndarray,
        task_input: EmbeddingTaskInput
    ) -> List[MetricResult]:
        """
        Run embedding task and compute metrics.
        
        Parameters
        ----------
        cell_representation : np.ndarray
            Cell embeddings, shape (n_cells, n_features)
        task_input : EmbeddingTaskInput
            Task input containing ground truth labels
            
        Returns
        -------
        List[MetricResult]
            List containing Silhouette score
        """
        logger.info(f"Starting EmbeddingTask with {cell_representation.shape[0]} cells, "
                   f"{cell_representation.shape[1]} features")
        
        labels = task_input.input_labels
        n_unique_labels = len(np.unique(labels))
        logger.debug(f"Computing silhouette score for {n_unique_labels} unique labels")
        
        sil_score = compute_silhouette(cell_representation, labels)
        logger.info(f"Silhouette score: {sil_score:.4f}")
        
        return [
            MetricResult(
                metric_type=MetricType.SILHOUETTE_SCORE,
                value=sil_score
            )
        ]


class MetadataLabelPredictionTask:
    """
    Task for predicting labels from embeddings using cross-validation.
    
    Evaluates multiple classifiers (Logistic Regression, KNN, Random Forest)
    using stratified k-fold cross-validation. Reports F1, accuracy, precision,
    and recall metrics averaged across folds.
    
    This implementation follows the czbenchmarks API for compatibility.
    
    Parameters
    ----------
    random_seed : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def run(
        self,
        cell_representation: np.ndarray,
        task_input: MetadataLabelPredictionTaskInput
    ) -> List[MetricResult]:
        """
        Run classification task and compute metrics.
        
        Parameters
        ----------
        cell_representation : np.ndarray
            Cell embeddings, shape (n_cells, n_features)
        task_input : MetadataLabelPredictionTaskInput
            Task input containing labels and cross-validation settings
            
        Returns
        -------
        List[MetricResult]
            List of metric results (F1, accuracy, precision, recall, AUROC) for each classifier
        """
        labels = task_input.labels
        
        logger.info(f"Starting MetadataLabelPredictionTask with {len(labels)} samples")
        
        # Filter classes with minimum size requirement
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info(f"Total classes before filtering: {len(unique_labels)}")
        logger.debug(f"Class distribution: {dict(zip(unique_labels, counts))}")
        
        valid_labels = unique_labels[counts >= task_input.min_class_size]
        logger.info(f"Total classes after filtering (min_class_size={task_input.min_class_size}): {len(valid_labels)}")
        
        # Create mask for valid samples
        mask = np.isin(labels, valid_labels)
        X_filtered = cell_representation[mask]
        y_filtered = labels[mask]
        
        logger.info(f"Samples after filtering: {X_filtered.shape[0]} / {cell_representation.shape[0]} "
                   f"({100 * X_filtered.shape[0] / cell_representation.shape[0]:.1f}%)")
        
        # Convert labels to categorical codes for sklearn
        if isinstance(y_filtered, pd.Series):
            y_cat = pd.Categorical(y_filtered)
        else:
            y_cat = pd.Categorical(y_filtered.astype(str))
        y_codes = y_cat.codes
        
        # Determine averaging strategy based on number of classes
        n_classes = len(np.unique(y_codes))
        if n_classes < 2:
            logger.warning(f"Only {n_classes} class(es) remaining after filtering. "
                          "Cannot perform cross-validation. Returning empty results.")
            return []
        
        average_strategy = 'binary' if n_classes == 2 else 'macro'
        logger.info(f"Found {n_classes} classes, using '{average_strategy}' averaging for metrics")
        
        # Setup cross-validation scorers
        scorers = {
            'f1': make_scorer(f1_score, average=average_strategy, zero_division=0),
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average=average_strategy, zero_division=0),
            'recall': make_scorer(recall_score, average=average_strategy, zero_division=0),
        }
        
        # Add AUROC for multi-class and binary classification
        if n_classes == 2:
            scorers['auroc'] = make_scorer(roc_auc_score, needs_proba=True)
        else:
            scorers['auroc'] = make_scorer(
                roc_auc_score,
                average='macro',
                multi_class='ovr',
                needs_proba=True
            )
        
        # Setup stratified k-fold
        skf = StratifiedKFold(
            n_splits=task_input.n_folds,
            shuffle=True,
            random_state=self.random_seed
        )
        logger.info(f"Using {task_input.n_folds}-fold stratified cross-validation with random_seed={self.random_seed}")
        
        # Define classifiers (using simplified versions)
        classifiers = {
            'lr': Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('lr', LogisticRegression(random_state=self.random_seed, max_iter=1000))
            ]),
            'knn': Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('knn', KNeighborsClassifier())
            ]),
            'rf': Pipeline([
                ('rf', RandomForestClassifier(random_state=self.random_seed, n_estimators=100))
            ])
        }
        logger.info(f"Created classifiers: {list(classifiers.keys())}")
        
        # Run cross-validation for each classifier and collect results
        all_results = []
        for clf_name, clf in classifiers.items():
            logger.info(f"Running cross-validation for {clf_name}...")
            
            cv_results = cross_validate(
                clf,
                X_filtered,
                y_codes,
                cv=skf,
                scoring=scorers,
                return_train_score=False
            )
            
            # Store mean scores across folds
            all_results.append(
                MetricResult(
                    metric_type=MetricType.MEAN_FOLD_F1_SCORE,
                    value=cv_results['test_f1'].mean(),
                    params={'classifier': clf_name}
                )
            )
            all_results.append(
                MetricResult(
                    metric_type=MetricType.MEAN_FOLD_ACCURACY,
                    value=cv_results['test_accuracy'].mean(),
                    params={'classifier': clf_name}
                )
            )
            all_results.append(
                MetricResult(
                    metric_type=MetricType.MEAN_FOLD_PRECISION,
                    value=cv_results['test_precision'].mean(),
                    params={'classifier': clf_name}
                )
            )
            all_results.append(
                MetricResult(
                    metric_type=MetricType.MEAN_FOLD_RECALL,
                    value=cv_results['test_recall'].mean(),
                    params={'classifier': clf_name}
                )
            )
            all_results.append(
                MetricResult(
                    metric_type=MetricType.MEAN_FOLD_AUROC,
                    value=cv_results['test_auroc'].mean(),
                    params={'classifier': clf_name}
                )
            )
            
            logger.debug(f"{clf_name} results: F1={cv_results['test_f1'].mean():.4f}, "
                        f"Accuracy={cv_results['test_accuracy'].mean():.4f}, "
                        f"Precision={cv_results['test_precision'].mean():.4f}, "
                        f"Recall={cv_results['test_recall'].mean():.4f}, "
                        f"AUROC={cv_results['test_auroc'].mean():.4f}")
        
        logger.info("Completed cross-validation for all classifiers")
        
        # Compute overall averages (across all classifiers)
        f1_scores = [r.value for r in all_results if r.metric_type == MetricType.MEAN_FOLD_F1_SCORE]
        accuracy_scores = [r.value for r in all_results if r.metric_type == MetricType.MEAN_FOLD_ACCURACY]
        precision_scores = [r.value for r in all_results if r.metric_type == MetricType.MEAN_FOLD_PRECISION]
        recall_scores = [r.value for r in all_results if r.metric_type == MetricType.MEAN_FOLD_RECALL]
        auroc_scores = [r.value for r in all_results if r.metric_type == MetricType.MEAN_FOLD_AUROC]
        
        # Add overall metrics (no classifier param)
        overall_results = [
            MetricResult(
                metric_type=MetricType.MEAN_FOLD_F1_SCORE,
                value=np.mean(f1_scores)
            ),
            MetricResult(
                metric_type=MetricType.MEAN_FOLD_ACCURACY,
                value=np.mean(accuracy_scores)
            ),
            MetricResult(
                metric_type=MetricType.MEAN_FOLD_PRECISION,
                value=np.mean(precision_scores)
            ),
            MetricResult(
                metric_type=MetricType.MEAN_FOLD_RECALL,
                value=np.mean(recall_scores)
            ),
            MetricResult(
                metric_type=MetricType.MEAN_FOLD_AUROC,
                value=np.mean(auroc_scores)
            ),
        ]
        
        logger.info(f"Overall metrics (averaged across all classifiers): "
                   f"F1={np.mean(f1_scores):.4f}, "
                   f"Accuracy={np.mean(accuracy_scores):.4f}, "
                   f"Precision={np.mean(precision_scores):.4f}, "
                   f"Recall={np.mean(recall_scores):.4f}, "
                   f"AUROC={np.mean(auroc_scores):.4f}")
        
        # Return overall metrics first, then per-classifier metrics
        return overall_results + all_results
