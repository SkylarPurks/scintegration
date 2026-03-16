"""
Pytest fixtures for scintegration tests.

Provides mock data in czbenchmarks format for testing.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock


# Mock MetricType enum (since we might not have czbenchmarks installed during tests)
class MockMetricType:
    ADJUSTED_RAND_INDEX = "adjusted_rand_index"
    NORMALIZED_MUTUAL_INFO = "normalized_mutual_info"
    SILHOUETTE_SCORE = "silhouette_score"
    MEAN_FOLD_F1_SCORE = "mean_fold_f1_score"
    MEAN_FOLD_ACCURACY = "mean_fold_accuracy"
    MEAN_FOLD_PRECISION = "mean_fold_precision"
    MEAN_FOLD_RECALL = "mean_fold_recall"
    MEAN_FOLD_AUROC = "mean_fold_auroc"


@pytest.fixture
def mock_metric_type(monkeypatch):
    """Mock czbenchmarks MetricType for testing."""
    # This will be used when testing evaluator
    import sys
    mock_module = MagicMock()
    mock_module.MetricType = MockMetricType
    sys.modules['czbenchmarks'] = MagicMock()
    sys.modules['czbenchmarks.metrics'] = MagicMock()
    sys.modules['czbenchmarks.metrics.types'] = mock_module
    return MockMetricType


@pytest.fixture
def sample_clustering_results_biology():
    """
    Sample clustering results for biological labels (cell types).
    Format matches czbenchmarks ClusteringTask output.
    """
    return {
        'scvi': {
            MockMetricType.ADJUSTED_RAND_INDEX: 0.732,
            MockMetricType.NORMALIZED_MUTUAL_INFO: 0.785,
            MockMetricType.SILHOUETTE_SCORE: 0.434,  # Raw value [-1, 1]
        },
        'scgpt': {
            MockMetricType.ADJUSTED_RAND_INDEX: 0.720,
            MockMetricType.NORMALIZED_MUTUAL_INFO: 0.774,
            MockMetricType.SILHOUETTE_SCORE: 0.398,
        },
        'pca': {
            MockMetricType.ADJUSTED_RAND_INDEX: 0.645,
            MockMetricType.NORMALIZED_MUTUAL_INFO: 0.698,
            MockMetricType.SILHOUETTE_SCORE: 0.312,
        }
    }


@pytest.fixture
def sample_clustering_results_batch():
    """
    Sample clustering results for batch labels (donors).
    """
    return {
        'scvi': {
            MockMetricType.ADJUSTED_RAND_INDEX: 0.605,
            MockMetricType.NORMALIZED_MUTUAL_INFO: 0.657,
            MockMetricType.SILHOUETTE_SCORE: 0.128,
        },
        'scgpt': {
            MockMetricType.ADJUSTED_RAND_INDEX: 0.620,
            MockMetricType.NORMALIZED_MUTUAL_INFO: 0.672,
            MockMetricType.SILHOUETTE_SCORE: 0.154,
        },
        'pca': {
            MockMetricType.ADJUSTED_RAND_INDEX: 0.712,
            MockMetricType.NORMALIZED_MUTUAL_INFO: 0.745,
            MockMetricType.SILHOUETTE_SCORE: 0.268,
        }
    }


@pytest.fixture
def sample_classification_results_biology():
    """
    Sample classification results for biological labels.
    Format matches czbenchmarks MetadataLabelPredictionTask output.
    """
    return {
        'scvi': {
            MockMetricType.MEAN_FOLD_F1_SCORE: 0.554,
            MockMetricType.MEAN_FOLD_ACCURACY: 0.863,
            MockMetricType.MEAN_FOLD_PRECISION: 0.578,
            MockMetricType.MEAN_FOLD_RECALL: 0.554,
        },
        'scgpt': {
            MockMetricType.MEAN_FOLD_F1_SCORE: 0.536,
            MockMetricType.MEAN_FOLD_ACCURACY: 0.851,
            MockMetricType.MEAN_FOLD_PRECISION: 0.562,
            MockMetricType.MEAN_FOLD_RECALL: 0.536,
        },
        'pca': {
            MockMetricType.MEAN_FOLD_F1_SCORE: 0.478,
            MockMetricType.MEAN_FOLD_ACCURACY: 0.812,
            MockMetricType.MEAN_FOLD_PRECISION: 0.502,
            MockMetricType.MEAN_FOLD_RECALL: 0.478,
        }
    }


@pytest.fixture
def sample_classification_results_batch():
    """
    Sample classification results for batch labels.
    """
    return {
        'scvi': {
            MockMetricType.MEAN_FOLD_F1_SCORE: 0.413,
            MockMetricType.MEAN_FOLD_ACCURACY: 0.667,
            MockMetricType.MEAN_FOLD_PRECISION: 0.445,
            MockMetricType.MEAN_FOLD_RECALL: 0.413,
        },
        'scgpt': {
            MockMetricType.MEAN_FOLD_F1_SCORE: 0.437,
            MockMetricType.MEAN_FOLD_ACCURACY: 0.689,
            MockMetricType.MEAN_FOLD_PRECISION: 0.468,
            MockMetricType.MEAN_FOLD_RECALL: 0.437,
        },
        'pca': {
            MockMetricType.MEAN_FOLD_F1_SCORE: 0.521,
            MockMetricType.MEAN_FOLD_ACCURACY: 0.745,
            MockMetricType.MEAN_FOLD_PRECISION: 0.548,
            MockMetricType.MEAN_FOLD_RECALL: 0.521,
        }
    }


@pytest.fixture
def expected_integration_scores():
    """
    Expected integration scores for the sample data above.
    These are pre-calculated to verify correctness.
    """
    # Recalculated correctly: IS = (B-L)/(2*(B+L))
    # scVI: (0.6970-0.5597)/(2*(0.6970+0.5597)) = 0.1373/2.5134 = 0.05463
    # scGPT: (0.6822-0.5764)/(2*(0.6822+0.5764)) = 0.1058/2.5172 = 0.04203
    # PCA: (0.6332-0.6867)/(2*(0.6332+0.6867)) = -0.0535/2.6398 = -0.02027
    return {
        'scvi': {
            'B': 0.6970,  # Average of ARI=0.732, NMI=0.785, Sil_norm=0.717, F1=0.554
            'L': 0.5597,  # Average of ARI=0.605, NMI=0.657, Sil_norm=0.564, F1=0.413
            'IntegrationScore': 0.05463,  # (B-L)/(2*(B+L))
        },
        'scgpt': {
            'B': 0.6822,
            'L': 0.5764,
            'IntegrationScore': 0.04203,
        },
        'pca': {
            'B': 0.6332,
            'L': 0.6867,
            'IntegrationScore': -0.02027,
        }
    }


@pytest.fixture
def edge_case_data():
    """Edge cases for testing robustness."""
    return {
        'zero_sum': {'B': 0.0, 'L': 0.0},
        'perfect_biology': {'B': 1.0, 'L': 0.0},
        'perfect_batch': {'B': 0.0, 'L': 1.0},
        'equal': {'B': 0.5, 'L': 0.5},
        'high_both': {'B': 0.9, 'L': 0.9},
    }


@pytest.fixture
def silhouette_test_values():
    """Test values for silhouette normalization."""
    return {
        'raw': np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        'normalized': np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    }
