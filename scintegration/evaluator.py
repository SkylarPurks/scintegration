"""
Main API for evaluating integration scores from task results.

This module provides the high-level interface for calculating integration scores
from clustering and classification task results.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from .tasks import MetricType
from .core import calculate_integration_score, compute_B_score, compute_L_score
from .normalization import normalize_silhouette, normalize_ari

logger = logging.getLogger(__name__)


@dataclass
class ModelScore:
    """
    Container for a single model's integration score and components.
    
    Attributes
    ----------
    model_name : str
        Name of the embedding model (e.g., 'scvi', 'scgpt')
    integration_score : float
        Final integration score [-0.5, 0.5]
    B : float
        Biology preservation score [0, 1]
    L : float
        Batch leakage score [0, 1]
    biology_components : dict
        Individual biology metrics (ari, nmi, silhouette, f1)
    batch_components : dict
        Individual batch metrics (ari, nmi, silhouette, f1)
    """
    model_name: str
    integration_score: float
    B: float
    L: float
    biology_components: Dict[str, float] = field(default_factory=dict)
    batch_components: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"ModelScore(model='{self.model_name}', "
            f"IS={self.integration_score:.4f}, "
            f"B={self.B:.4f}, L={self.L:.4f})"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'model_name': self.model_name,
            'integration_score': self.integration_score,
            'B': self.B,
            'L': self.L,
            'biology_components': self.biology_components,
            'batch_components': self.batch_components
        }


class IntegrationScoreResults:
    """
    Container for integration score results across multiple models.
    
    Attributes
    ----------
    scores : dict
        Dictionary mapping model names to ModelScore objects
    model_names : list
        List of evaluated model names
    """
    
    def __init__(self, scores: Dict[str, ModelScore]):
        """
        Initialize results container.
        
        Parameters
        ----------
        scores : dict
            Dictionary mapping model names to ModelScore objects
        """
        self.scores = scores
        self.model_names = list(scores.keys())
    
    @property
    def best_model(self) -> str:
        """Get name of model with highest integration score."""
        return max(self.scores.items(), key=lambda x: x[1].integration_score)[0]
    
    @property
    def best_score(self) -> ModelScore:
        """Get ModelScore object for best performing model."""
        return self.scores[self.best_model]
    
    def get_ranked_models(self) -> List[tuple]:
        """
        Get models ranked by integration score (highest to lowest).
        
        Returns
        -------
        list of tuples
            List of (model_name, ModelScore) tuples sorted by integration score
        """
        return sorted(
            self.scores.items(),
            key=lambda x: x[1].integration_score,
            reverse=True
        )
    
    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: model_name, integration_score, B, L,
            and all component metrics
        
        Raises
        ------
        ImportError
            If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )
        
        data = []
        for model_name, score in self.scores.items():
            row = {
                'model_name': model_name,
                'integration_score': score.integration_score,
                'B': score.B,
                'L': score.L,
                **{f'bio_{k}': v for k, v in score.biology_components.items()},
                **{f'batch_{k}': v for k, v in score.batch_components.items()}
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def summary(self) -> str:
        """
        Generate a text summary of results.
        
        Returns
        -------
        str
            Formatted summary string
        """
        lines = ["=" * 80]
        lines.append("INTEGRATION SCORE RESULTS")
        lines.append("=" * 80)
        
        for rank, (model_name, score) in enumerate(self.get_ranked_models(), 1):
            lines.append(f"\n#{rank} {model_name.upper()}:")
            lines.append(f"  Integration Score: {score.integration_score:+.4f}")
            lines.append(f"  Biology (B):       {score.B:.4f}")
            lines.append(f"  Leakage (L):       {score.L:.4f}")
            
            bio = score.biology_components
            lines.append(
                f"    Biology metrics: "
                f"ARI={bio['ari']:.3f}, NMI={bio['nmi']:.3f}, "
                f"Sil={bio['silhouette']:.3f}, F1={bio['f1']:.3f}"
            )
            
            batch = score.batch_components
            # Handle case where F1 is not available (single batch)
            if 'f1' in batch:
                lines.append(
                    f"    Batch metrics:   "
                    f"ARI={batch['ari']:.3f}, NMI={batch['nmi']:.3f}, "
                    f"Sil={batch['silhouette']:.3f}, F1={batch['f1']:.3f}"
                )
            else:
                lines.append(
                    f"    Batch metrics:   "
                    f"ARI={batch['ari']:.3f}, NMI={batch['nmi']:.3f}, "
                    f"Sil={batch['silhouette']:.3f}, F1=N/A (single batch)"
                )
        
        lines.append("\n" + "=" * 80)
        lines.append(f"BEST MODEL: {self.best_model.upper()} "
                    f"(IS = {self.best_score.integration_score:+.4f})")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def to_csv(self, output_dir: str = ".", filename: Optional[str] = None) -> str:
        """
        Export results to CSV file with automatic naming based on date.
        
        This is a convenience method that converts the results to a DataFrame
        and exports it to CSV with a filename that includes the current date and time.
        
        Parameters
        ----------
        output_dir : str, default="."
            Directory where the CSV file will be saved. Will be created if it doesn't exist.
        filename : str, optional
            Custom filename for the CSV file. If provided, overrides automatic naming.
            Otherwise, uses format: integration_metrics_all_models_{date}_{time}.csv
        
        Returns
        -------
        str
            Path to the saved CSV file
        
        Examples
        --------
        >>> from scintegration import IntegrationScoreEvaluator
        >>> 
        >>> # Evaluate models
        >>> evaluator = IntegrationScoreEvaluator()
        >>> results = evaluator.evaluate(...)
        >>> 
        >>> # Export to CSV in current directory
        >>> csv_path = results.to_csv()
        >>> print(f"Saved to: {csv_path}")
        
        >>> # Export to specific directory
        >>> csv_path = results.to_csv(output_dir="results/experiments")
        
        >>> # Use custom filename
        >>> csv_path = results.to_csv(filename="my_experiment_metrics.csv")
        
        See Also
        --------
        to_dataframe : Convert results to pandas DataFrame
        export_metrics_to_csv : Low-level function for exporting DataFrames
        """
        from .utils import export_metrics_to_csv
        
        # Convert to DataFrame
        df = self.to_dataframe()
        
        # Export using utility function
        return export_metrics_to_csv(
            data=df,
            model_name=None,  # Use "all_models" in filename
            output_dir=output_dir,
            filename=filename
        )
    
    def __repr__(self) -> str:
        return f"IntegrationScoreResults({len(self.scores)} models, best={self.best_model})"


class IntegrationScoreEvaluator:
    """
    Calculate integration scores from task results.
    
    This evaluator takes the output of ClusteringTask, EmbeddingTask, and
    MetadataLabelPredictionTask and computes integration scores that balance
    biology preservation against batch leakage.
    
    Parameters
    ----------
    clustering_metrics : list of str, default=['ari', 'nmi', 'silhouette']
        Which clustering metrics to include in B and L calculation
    classification_metrics : list of str, default=['f1']
        Which classification metrics to include in B and L calculation
        Options: 'f1', 'accuracy', 'precision', 'recall', 'auroc'
    weights : str or dict, default='equal'
        Weighting scheme for metrics. 'equal' or custom dict.
    
    Examples
    --------
    >>> from scintegration import IntegrationScoreEvaluator
    >>> from scintegration import ClusteringTask, MetadataLabelPredictionTask
    >>> 
    >>> # Run tasks
    >>> clustering_task_bio = ClusteringTask()
    >>> clustering_results_bio = clustering_task_bio.evaluate(...)
    >>> # ... (run other tasks)
    >>> 
    >>> # Calculate integration scores
    >>> evaluator = IntegrationScoreEvaluator()
    >>> results = evaluator.evaluate(
    ...     clustering_results_biology=clustering_results_bio,
    ...     clustering_results_batch=clustering_results_batch,
    ...     classification_results_biology=classification_results_bio,
    ...     classification_results_batch=classification_results_batch
    ... )
    >>> 
    >>> print(results.summary())
    >>> print(f"Best model: {results.best_model}")
    """
    
    def __init__(
        self,
        clustering_metrics: List[str] = None,
        classification_metrics: List[str] = None,
        weights: Union[str, dict] = 'equal'
    ):
        self.clustering_metrics = clustering_metrics or ['ari', 'nmi', 'silhouette']
        self.classification_metrics = classification_metrics or ['f1']
        self.weights = weights
        
        # Validate metrics
        valid_clustering = {'ari', 'nmi', 'silhouette'}
        valid_classification = {'f1', 'accuracy', 'precision', 'recall', 'auroc'}
        
        for metric in self.clustering_metrics:
            if metric not in valid_clustering:
                raise ValueError(
                    f"Invalid clustering metric: {metric}. "
                    f"Must be one of {valid_clustering}"
                )
        
        for metric in self.classification_metrics:
            if metric not in valid_classification:
                raise ValueError(
                    f"Invalid classification metric: {metric}. "
                    f"Must be one of {valid_classification}"
                )
    
    def evaluate(
        self,
        clustering_results_biology: Dict,
        clustering_results_batch: Dict,
        classification_results_biology: Dict,
        classification_results_batch: Dict
    ) -> IntegrationScoreResults:
        """
        Calculate integration scores from task results.
        
        Parameters
        ----------
        clustering_results_biology : dict
            Results from ClusteringTask with biological labels
            Format: {model_name: {MetricType.ADJUSTED_RAND_INDEX: value, ...}}
        clustering_results_batch : dict
            Results from ClusteringTask with batch labels
        classification_results_biology : dict
            Results from MetadataLabelPredictionTask for biology
            Format: {model_name: {MetricType.MEAN_FOLD_F1_SCORE: value, ...}}
        classification_results_batch : dict
            Results from MetadataLabelPredictionTask for batch
        
        Returns
        -------
        IntegrationScoreResults
            Object containing integration scores for all models
        
        Raises
        ------
        ValueError
            If model names don't match across result dictionaries
        KeyError
            If required metrics are missing from results
        """
        # Validate that all result dicts have the same models
        model_sets = [
            set(clustering_results_biology.keys()),
            set(clustering_results_batch.keys()),
            set(classification_results_biology.keys()),
            set(classification_results_batch.keys())
        ]
        
        if not all(models == model_sets[0] for models in model_sets):
            raise ValueError(
                "Model names must match across all result dictionaries. "
                f"Found: {[sorted(m) for m in model_sets]}"
            )
        
        scores = {}
        
        # Check if batch F1 is missing (indicating single batch scenario)
        first_model = list(clustering_results_biology.keys())[0]
        if MetricType.MEAN_FOLD_F1_SCORE not in classification_results_batch[first_model]:
            warnings.warn(
                "Batch F1 classification metric not found. This typically indicates single-batch data. "
                "Integration scores calculated without batch classification may not be meaningful. "
                "Consider using data with multiple batches for proper integration assessment.",
                UserWarning,
                stacklevel=2
            )
            logger.warning("Batch F1 not found - likely single batch scenario")
        
        for model_name in clustering_results_biology.keys():
            # Extract biology metrics
            ari_bio_raw = clustering_results_biology[model_name][MetricType.ADJUSTED_RAND_INDEX]
            ari_bio = normalize_ari(ari_bio_raw)
            nmi_bio = clustering_results_biology[model_name][MetricType.NORMALIZED_MUTUAL_INFO]
            sil_bio_raw = clustering_results_biology[model_name][MetricType.SILHOUETTE_SCORE]
            sil_bio = normalize_silhouette(sil_bio_raw)
            # F1 might not be available for rare cases
            f1_bio = classification_results_biology[model_name].get(MetricType.MEAN_FOLD_F1_SCORE, 0.0)
            
            # Extract batch metrics
            ari_batch_raw = clustering_results_batch[model_name][MetricType.ADJUSTED_RAND_INDEX]
            ari_batch = normalize_ari(ari_batch_raw)
            nmi_batch = clustering_results_batch[model_name][MetricType.NORMALIZED_MUTUAL_INFO]
            sil_batch_raw = clustering_results_batch[model_name][MetricType.SILHOUETTE_SCORE]
            sil_batch = normalize_silhouette(sil_batch_raw)
            # F1 might not be available for batch with very few classes (2-3 donors) or single batch
            f1_batch = classification_results_batch[model_name].get(MetricType.MEAN_FOLD_F1_SCORE)
            # Use None if F1 is not available (single batch case)
            if f1_batch is None:
                f1_batch_for_score = None
            else:
                f1_batch_for_score = f1_batch
            
            # Compute B and L scores
            B = compute_B_score(ari_bio, nmi_bio, sil_bio, f1_bio, weights=self.weights)
            L = compute_L_score(ari_batch, nmi_batch, sil_batch, f1_batch_for_score, weights=self.weights)
            
            # Compute integration score
            IS = calculate_integration_score(B, L)
            
            # Extract ALL biology classification metrics (for export)
            bio_components = {
                'ari': ari_bio,  # normalized [0, 1]
                'nmi': nmi_bio,  # already [0, 1]
                'silhouette': sil_bio,  # normalized [0, 1]
                'f1': f1_bio  # already [0, 1]
            }
            # Add additional classification metrics if available
            if MetricType.MEAN_FOLD_ACCURACY in classification_results_biology[model_name]:
                bio_components['accuracy'] = classification_results_biology[model_name][MetricType.MEAN_FOLD_ACCURACY]
            if MetricType.MEAN_FOLD_PRECISION in classification_results_biology[model_name]:
                bio_components['precision'] = classification_results_biology[model_name][MetricType.MEAN_FOLD_PRECISION]
            if MetricType.MEAN_FOLD_RECALL in classification_results_biology[model_name]:
                bio_components['recall'] = classification_results_biology[model_name][MetricType.MEAN_FOLD_RECALL]
            if MetricType.MEAN_FOLD_AUROC in classification_results_biology[model_name]:
                bio_components['auroc'] = classification_results_biology[model_name][MetricType.MEAN_FOLD_AUROC]
            
            # Extract ALL batch classification metrics (for export)
            batch_components = {
                'ari': ari_batch,  # normalized [0, 1]
                'nmi': nmi_batch,  # already [0, 1]
                'silhouette': sil_batch,  # normalized [0, 1]
            }
            # Add F1 only if available (excluded for single batch case)
            if f1_batch is not None:
                batch_components['f1'] = f1_batch  # already [0, 1]
            # Add additional classification metrics if available
            if MetricType.MEAN_FOLD_ACCURACY in classification_results_batch[model_name]:
                batch_components['accuracy'] = classification_results_batch[model_name][MetricType.MEAN_FOLD_ACCURACY]
            if MetricType.MEAN_FOLD_PRECISION in classification_results_batch[model_name]:
                batch_components['precision'] = classification_results_batch[model_name][MetricType.MEAN_FOLD_PRECISION]
            if MetricType.MEAN_FOLD_RECALL in classification_results_batch[model_name]:
                batch_components['recall'] = classification_results_batch[model_name][MetricType.MEAN_FOLD_RECALL]
            if MetricType.MEAN_FOLD_AUROC in classification_results_batch[model_name]:
                batch_components['auroc'] = classification_results_batch[model_name][MetricType.MEAN_FOLD_AUROC]
            
            # Store results
            scores[model_name] = ModelScore(
                model_name=model_name,
                integration_score=IS,
                B=B,
                L=L,
                biology_components=bio_components,
                batch_components=batch_components
            )
        
        return IntegrationScoreResults(scores)
    
    def evaluate_embeddings(
        self,
        embeddings: np.ndarray,
        obs: pd.DataFrame,
        biology_labels: np.ndarray,
        batch_labels: np.ndarray,
        model_name: str = 'model',
        classification_n_folds: Optional[int] = None,
        classification_min_class_size: Optional[int] = None,
        clustering_n_neighbors: Optional[int] = None,
        clustering_resolution: Optional[float] = None
    ) -> IntegrationScoreResults:
        """
        Convenience method to evaluate embeddings directly.
        
        This method handles all task execution internally, including:
        - Clustering metrics (ARI, NMI)
        - Embedding metrics (Silhouette)
        - Classification metrics (F1, Accuracy, etc.)
        
        Parameters
        ----------
        embeddings : np.ndarray
            Cell embeddings of shape (n_cells, n_features)
        obs : pd.DataFrame
            AnnData observation metadata
        biology_labels : np.ndarray
            Biological labels (e.g., cell types) to preserve
        batch_labels : np.ndarray
            Batch labels (e.g., donors) to remove
        model_name : str, default='model'
            Name for this embedding/model
        classification_n_folds : int, optional
            Number of cross-validation folds for classification tasks.
            If None, uses task default (5). Increase for more robust estimates
            or decrease for faster computation.
        classification_min_class_size : int, optional
            Minimum number of samples required per class. Classes with fewer
            samples will be filtered out. If None, uses task default (10).
            Decrease to include rare cell types or increase for more stable estimates.
        clustering_n_neighbors : int, optional
            Number of neighbors for k-NN graph in clustering. If None, uses
            task default (15). Higher values create more connected graphs,
            leading to larger, smoother clusters.
        clustering_resolution : float, optional
            Resolution parameter for Leiden clustering. If None, uses task
            default (1.0). Higher values yield more fine-grained clusters,
            lower values produce coarser clusters.
        
        Returns
        -------
        IntegrationScoreResults
            Integration score results
        
        Examples
        --------
        >>> # Basic usage with defaults
        >>> evaluator = IntegrationScoreEvaluator()
        >>> results = evaluator.evaluate_embeddings(
        ...     embeddings=scvi_embeddings,
        ...     obs=adata.obs,
        ...     biology_labels=cell_type_labels,
        ...     batch_labels=donor_labels,
        ...     model_name='scvi'
        ... )
        >>> print(results.summary())
        >>> 
        >>> # Advanced: custom parameters for rare cell types and clustering
        >>> results = evaluator.evaluate_embeddings(
        ...     embeddings=scvi_embeddings,
        ...     obs=adata.obs,
        ...     biology_labels=cell_type_labels,
        ...     batch_labels=donor_labels,
        ...     model_name='scvi',
        ...     classification_n_folds=10,  # More robust estimates
        ...     classification_min_class_size=3,  # Include rare cell types
        ...     clustering_n_neighbors=20,  # More connected graph
        ...     clustering_resolution=0.5  # Coarser clusters
        ... )
        """
        from .tasks import (
            ClusteringTask, EmbeddingTask, MetadataLabelPredictionTask,
            ClusteringTaskInput, EmbeddingTaskInput, MetadataLabelPredictionTaskInput
        )
        
        # Check if there's only one batch
        n_unique_batches = len(np.unique(batch_labels))
        
        if n_unique_batches == 1:
            warnings.warn(
                f"Only 1 unique batch detected in batch_labels. "
                f"Integration scores are not meaningful for single-batch data. "
                f"The batch leakage (L) score will be calculated using only clustering metrics (ARI, NMI, Silhouette), "
                f"and F1 classification will be excluded. "
                f"For meaningful integration assessment, provide data with multiple batches (e.g., multiple donors, sequencing runs, or technical replicates).",
                UserWarning,
                stacklevel=2
            )
            logger.warning(f"Single batch detected - integration score may not be meaningful")
        
        # Initialize tasks with optional custom parameters
        clustering_kwargs = {}
        if clustering_n_neighbors is not None:
            clustering_kwargs['n_neighbors'] = clustering_n_neighbors
        if clustering_resolution is not None:
            clustering_kwargs['resolution'] = clustering_resolution
        
        clustering_task = ClusteringTask(**clustering_kwargs)
        embedding_task = EmbeddingTask()
        classification_task = MetadataLabelPredictionTask()
        
        # Evaluate biology preservation
        # Clustering metrics (ARI, NMI)
        clustering_input_bio = ClusteringTaskInput(obs=obs, input_labels=biology_labels)
        clustering_results_bio = clustering_task.run(
            cell_representation=embeddings,
            task_input=clustering_input_bio
        )
        clustering_bio = {model_name: {r.metric_type: r.value for r in clustering_results_bio}}
        
        # Silhouette score
        embedding_task_input_bio = EmbeddingTaskInput(input_labels=biology_labels)
        embedding_results_bio = embedding_task.run(
            cell_representation=embeddings,
            task_input=embedding_task_input_bio
        )
        for r in embedding_results_bio:
            if r.metric_type == MetricType.SILHOUETTE_SCORE:
                clustering_bio[model_name][MetricType.SILHOUETTE_SCORE] = r.value
        
        # Classification metrics
        # Build classification input with optional custom parameters
        classification_kwargs = {'labels': biology_labels}
        if classification_n_folds is not None:
            classification_kwargs['n_folds'] = classification_n_folds
        if classification_min_class_size is not None:
            classification_kwargs['min_class_size'] = classification_min_class_size
        
        classification_input_bio = MetadataLabelPredictionTaskInput(**classification_kwargs)
        classification_results_bio = classification_task.run(
            cell_representation=embeddings,
            task_input=classification_input_bio
        )
        classification_bio = {model_name: {r.metric_type: r.value for r in classification_results_bio}}
        
        # Evaluate batch removal
        # Clustering metrics (ARI, NMI)
        clustering_input_batch = ClusteringTaskInput(obs=obs, input_labels=batch_labels)
        clustering_results_batch = clustering_task.run(
            cell_representation=embeddings,
            task_input=clustering_input_batch
        )
        clustering_batch = {model_name: {r.metric_type: r.value for r in clustering_results_batch}}
        
        # Silhouette score
        embedding_task_input_batch = EmbeddingTaskInput(input_labels=batch_labels)
        embedding_results_batch = embedding_task.run(
            cell_representation=embeddings,
            task_input=embedding_task_input_batch
        )
        for r in embedding_results_batch:
            if r.metric_type == MetricType.SILHOUETTE_SCORE:
                clustering_batch[model_name][MetricType.SILHOUETTE_SCORE] = r.value
        
        # Classification metrics (only if more than one batch)
        classification_batch = {model_name: {}}
        if n_unique_batches > 1:
            # Build classification input with optional custom parameters
            classification_kwargs_batch = {'labels': batch_labels}
            if classification_n_folds is not None:
                classification_kwargs_batch['n_folds'] = classification_n_folds
            if classification_min_class_size is not None:
                classification_kwargs_batch['min_class_size'] = classification_min_class_size
            
            classification_input_batch = MetadataLabelPredictionTaskInput(**classification_kwargs_batch)
            classification_results_batch = classification_task.run(
                cell_representation=embeddings,
                task_input=classification_input_batch
            )
            classification_batch = {model_name: {r.metric_type: r.value for r in classification_results_batch}}
        
        # Calculate integration score
        return self.evaluate(
            clustering_results_biology=clustering_bio,
            clustering_results_batch=clustering_batch,
            classification_results_biology=classification_bio,
            classification_results_batch=classification_batch
        )
