"""
Tests for IntegrationScoreEvaluator and related classes.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scintegration.evaluator import (
    IntegrationScoreEvaluator,
    IntegrationScoreResults,
    ModelScore
)
from tests.conftest import MockMetricType


class TestModelScore:
    """Tests for ModelScore dataclass."""
    
    def test_creation(self):
        """Test creating a ModelScore object."""
        score = ModelScore(
            model_name='scvi',
            integration_score=0.0702,
            B=0.6970,
            L=0.5597,
            biology_components={'ari': 0.732, 'nmi': 0.785, 'silhouette': 0.717, 'f1': 0.554},
            batch_components={'ari': 0.605, 'nmi': 0.657, 'silhouette': 0.564, 'f1': 0.413}
        )
        
        assert score.model_name == 'scvi'
        assert score.integration_score == pytest.approx(0.0702, abs=1e-4)
        assert score.B == pytest.approx(0.6970, abs=1e-4)
        assert score.L == pytest.approx(0.5597, abs=1e-4)
    
    def test_repr(self):
        """Test __repr__ method."""
        score = ModelScore(
            model_name='test',
            integration_score=0.1234,
            B=0.7,
            L=0.5
        )
        repr_str = repr(score)
        assert 'test' in repr_str
        assert '0.1234' in repr_str
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = ModelScore(
            model_name='scvi',
            integration_score=0.0702,
            B=0.6970,
            L=0.5597,
            biology_components={'ari': 0.732},
            batch_components={'ari': 0.605}
        )
        
        result = score.to_dict()
        assert isinstance(result, dict)
        assert result['model_name'] == 'scvi'
        assert result['integration_score'] == 0.0702
        assert 'biology_components' in result
        assert 'batch_components' in result


class TestIntegrationScoreResults:
    """Tests for IntegrationScoreResults class."""
    
    def test_creation(self):
        """Test creating results object."""
        scores = {
            'scvi': ModelScore('scvi', 0.07, 0.70, 0.56),
            'pca': ModelScore('pca', -0.03, 0.63, 0.69)
        }
        results = IntegrationScoreResults(scores)
        
        assert len(results.model_names) == 2
        assert 'scvi' in results.model_names
        assert 'pca' in results.model_names
    
    def test_best_model(self):
        """Test best_model property."""
        scores = {
            'scvi': ModelScore('scvi', 0.07, 0.70, 0.56),
            'scgpt': ModelScore('scgpt', 0.05, 0.68, 0.58),
            'pca': ModelScore('pca', -0.03, 0.63, 0.69)
        }
        results = IntegrationScoreResults(scores)
        
        assert results.best_model == 'scvi'
        assert results.best_score.integration_score == 0.07
    
    def test_get_ranked_models(self):
        """Test get_ranked_models method."""
        scores = {
            'pca': ModelScore('pca', -0.03, 0.63, 0.69),
            'scvi': ModelScore('scvi', 0.07, 0.70, 0.56),
            'scgpt': ModelScore('scgpt', 0.05, 0.68, 0.58),
        }
        results = IntegrationScoreResults(scores)
        
        ranked = results.get_ranked_models()
        assert len(ranked) == 3
        assert ranked[0][0] == 'scvi'  # Highest score
        assert ranked[1][0] == 'scgpt'
        assert ranked[2][0] == 'pca'  # Lowest score
    
    def test_summary(self):
        """Test summary method."""
        scores = {
            'scvi': ModelScore(
                'scvi', 0.0702, 0.6970, 0.5597,
                {'ari': 0.732, 'nmi': 0.785, 'silhouette': 0.717, 'f1': 0.554},
                {'ari': 0.605, 'nmi': 0.657, 'silhouette': 0.564, 'f1': 0.413}
            )
        }
        results = IntegrationScoreResults(scores)
        
        summary = results.summary()
        assert isinstance(summary, str)
        assert 'scvi' in summary.lower()
        assert '0.0702' in summary or '0.070' in summary
        assert 'BEST MODEL' in summary
    
    def test_repr(self):
        """Test __repr__ method."""
        scores = {
            'scvi': ModelScore('scvi', 0.07, 0.70, 0.56),
            'pca': ModelScore('pca', -0.03, 0.63, 0.69)
        }
        results = IntegrationScoreResults(scores)
        
        repr_str = repr(results)
        assert '2 models' in repr_str
        assert 'best=scvi' in repr_str


class TestIntegrationScoreEvaluator:
    """Tests for IntegrationScoreEvaluator class."""
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_initialization_default(self):
        """Test evaluator initialization with defaults."""
        evaluator = IntegrationScoreEvaluator()
        
        assert evaluator.clustering_metrics == ['ari', 'nmi', 'silhouette']
        assert evaluator.classification_metrics == ['f1']
        assert evaluator.weights == 'equal'
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_initialization_custom_metrics(self):
        """Test evaluator initialization with custom metrics."""
        evaluator = IntegrationScoreEvaluator(
            clustering_metrics=['ari', 'nmi'],
            classification_metrics=['f1', 'accuracy']
        )
        
        assert evaluator.clustering_metrics == ['ari', 'nmi']
        assert evaluator.classification_metrics == ['f1', 'accuracy']
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_invalid_clustering_metric(self):
        """Test that invalid clustering metric raises error."""
        with pytest.raises(ValueError, match="Invalid clustering metric"):
            IntegrationScoreEvaluator(clustering_metrics=['invalid_metric'])
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_invalid_classification_metric(self):
        """Test that invalid classification metric raises error."""
        with pytest.raises(ValueError, match="Invalid classification metric"):
            IntegrationScoreEvaluator(classification_metrics=['invalid_metric'])
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_evaluate_basic(
        self,
        sample_clustering_results_biology,
        sample_clustering_results_batch,
        sample_classification_results_biology,
        sample_classification_results_batch
    ):
        """Test basic evaluation with sample data."""
        evaluator = IntegrationScoreEvaluator()
        
        results = evaluator.evaluate(
            clustering_results_biology=sample_clustering_results_biology,
            clustering_results_batch=sample_clustering_results_batch,
            classification_results_biology=sample_classification_results_biology,
            classification_results_batch=sample_classification_results_batch
        )
        
        assert isinstance(results, IntegrationScoreResults)
        assert len(results.model_names) == 3
        assert 'scvi' in results.model_names
        assert 'scgpt' in results.model_names
        assert 'pca' in results.model_names
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_evaluate_scvi_scores(
        self,
        sample_clustering_results_biology,
        sample_clustering_results_batch,
        sample_classification_results_biology,
        sample_classification_results_batch,
        expected_integration_scores
    ):
        """Test that scVI scores match expected values."""
        evaluator = IntegrationScoreEvaluator()
        
        results = evaluator.evaluate(
            clustering_results_biology=sample_clustering_results_biology,
            clustering_results_batch=sample_clustering_results_batch,
            classification_results_biology=sample_classification_results_biology,
            classification_results_batch=sample_classification_results_batch
        )
        
        scvi_score = results.scores['scvi']
        expected = expected_integration_scores['scvi']
        
        assert scvi_score.B == pytest.approx(expected['B'], abs=1e-4)
        assert scvi_score.L == pytest.approx(expected['L'], abs=1e-4)
        assert scvi_score.integration_score == pytest.approx(expected['IntegrationScore'], abs=1e-4)
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_evaluate_best_model(
        self,
        sample_clustering_results_biology,
        sample_clustering_results_batch,
        sample_classification_results_biology,
        sample_classification_results_batch
    ):
        """Test that best model is correctly identified."""
        evaluator = IntegrationScoreEvaluator()
        
        results = evaluator.evaluate(
            clustering_results_biology=sample_clustering_results_biology,
            clustering_results_batch=sample_clustering_results_batch,
            classification_results_biology=sample_classification_results_biology,
            classification_results_batch=sample_classification_results_batch
        )
        
        # scVI should be best (highest integration score)
        assert results.best_model == 'scvi'
        assert results.best_score.integration_score > 0
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_evaluate_component_storage(
        self,
        sample_clustering_results_biology,
        sample_clustering_results_batch,
        sample_classification_results_biology,
        sample_classification_results_batch
    ):
        """Test that individual components are stored correctly."""
        evaluator = IntegrationScoreEvaluator()
        
        results = evaluator.evaluate(
            clustering_results_biology=sample_clustering_results_biology,
            clustering_results_batch=sample_clustering_results_batch,
            classification_results_biology=sample_classification_results_biology,
            classification_results_batch=sample_classification_results_batch
        )
        
        scvi_score = results.scores['scvi']
        
        # Check biology components
        assert 'ari' in scvi_score.biology_components
        assert 'nmi' in scvi_score.biology_components
        assert 'silhouette' in scvi_score.biology_components
        assert 'f1' in scvi_score.biology_components
        
        # Check batch components
        assert 'ari' in scvi_score.batch_components
        assert 'nmi' in scvi_score.batch_components
        assert 'silhouette' in scvi_score.batch_components
        assert 'f1' in scvi_score.batch_components
        
        # Verify actual values
        assert scvi_score.biology_components['ari'] == pytest.approx(0.732, abs=1e-3)
        assert scvi_score.biology_components['f1'] == pytest.approx(0.554, abs=1e-3)
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_mismatched_model_names(self):
        """Test that mismatched model names raise error."""
        clustering_bio = {
            'scvi': {MockMetricType.ADJUSTED_RAND_INDEX: 0.7}
        }
        clustering_batch = {
            'scgpt': {MockMetricType.ADJUSTED_RAND_INDEX: 0.6}  # Different model
        }
        classification_bio = {
            'scvi': {MockMetricType.MEAN_FOLD_F1_SCORE: 0.5}
        }
        classification_batch = {
            'scvi': {MockMetricType.MEAN_FOLD_F1_SCORE: 0.4}
        }
        
        evaluator = IntegrationScoreEvaluator()
        
        with pytest.raises(ValueError, match="Model names must match"):
            evaluator.evaluate(
                clustering_results_biology=clustering_bio,
                clustering_results_batch=clustering_batch,
                classification_results_biology=classification_bio,
                classification_results_batch=classification_batch
            )
    
    @patch('scintegration.evaluator.MetricType', MockMetricType)
    def test_silhouette_normalization(
        self,
        sample_clustering_results_biology,
        sample_clustering_results_batch,
        sample_classification_results_biology,
        sample_classification_results_batch
    ):
        """Test that silhouette scores are properly normalized."""
        evaluator = IntegrationScoreEvaluator()
        
        results = evaluator.evaluate(
            clustering_results_biology=sample_clustering_results_biology,
            clustering_results_batch=sample_clustering_results_batch,
            classification_results_biology=sample_classification_results_biology,
            classification_results_batch=sample_classification_results_batch
        )
        
        scvi_score = results.scores['scvi']
        
        # Raw silhouette for scVI bio was 0.434, should be normalized to (0.434+1)/2 = 0.717
        assert scvi_score.biology_components['silhouette'] == pytest.approx(0.717, abs=1e-3)
        
        # Raw silhouette for scVI batch was 0.128, should be normalized to (0.128+1)/2 = 0.564
        assert scvi_score.batch_components['silhouette'] == pytest.approx(0.564, abs=1e-3)
        
        # Normalized values should be in [0, 1]
        assert 0.0 <= scvi_score.biology_components['silhouette'] <= 1.0
        assert 0.0 <= scvi_score.batch_components['silhouette'] <= 1.0


class TestIntegrationScoreResultsDataFrame:
    """Tests for DataFrame conversion (when pandas is available)."""
    
    def test_to_dataframe_structure(self):
        """Test DataFrame structure."""
        pytest.importorskip("pandas")
        
        scores = {
            'scvi': ModelScore(
                'scvi', 0.0702, 0.6970, 0.5597,
                {'ari': 0.732, 'nmi': 0.785, 'silhouette': 0.717, 'f1': 0.554},
                {'ari': 0.605, 'nmi': 0.657, 'silhouette': 0.564, 'f1': 0.413}
            )
        }
        results = IntegrationScoreResults(scores)
        
        df = results.to_dataframe()
        
        assert 'model_name' in df.columns
        assert 'integration_score' in df.columns
        assert 'B' in df.columns
        assert 'L' in df.columns
        assert 'bio_ari' in df.columns
        assert 'batch_f1' in df.columns
        
        assert len(df) == 1
        assert df.loc[0, 'model_name'] == 'scvi'
    
    def test_to_dataframe_without_pandas(self):
        """Test that ImportError is raised without pandas."""
        scores = {
            'scvi': ModelScore('scvi', 0.07, 0.70, 0.56)
        }
        results = IntegrationScoreResults(scores)
        
        with patch.dict('sys.modules', {'pandas': None}):
            with pytest.raises(ImportError, match="pandas is required"):
                results.to_dataframe()
