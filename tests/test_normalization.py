"""
Tests for normalization utilities.
"""

import pytest
import numpy as np
from scintegration.normalization import normalize_silhouette, denormalize_silhouette


class TestNormalizeSilhouette:
    """Tests for normalize_silhouette function."""
    
    def test_perfect_clustering(self):
        """Test normalization of perfect silhouette score (1.0)."""
        assert normalize_silhouette(1.0) == 1.0
    
    def test_worst_clustering(self):
        """Test normalization of worst silhouette score (-1.0)."""
        assert normalize_silhouette(-1.0) == 0.0
    
    def test_neutral_clustering(self):
        """Test normalization of neutral silhouette score (0.0)."""
        assert normalize_silhouette(0.0) == 0.5
    
    def test_positive_score(self):
        """Test normalization of positive silhouette score."""
        assert normalize_silhouette(0.5) == 0.75
    
    def test_negative_score(self):
        """Test normalization of negative silhouette score."""
        assert normalize_silhouette(-0.5) == 0.25
    
    def test_array_input(self, silhouette_test_values):
        """Test normalization with numpy array input."""
        raw = silhouette_test_values['raw']
        expected = silhouette_test_values['normalized']
        result = normalize_silhouette(raw)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_output_range(self):
        """Test that output is always in [0, 1] range."""
        test_values = np.linspace(-1.0, 1.0, 50)
        normalized = normalize_silhouette(test_values)
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)
    
    def test_monotonicity(self):
        """Test that normalization preserves ordering."""
        values = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
        normalized = normalize_silhouette(values)
        # Check strictly increasing
        assert np.all(normalized[:-1] < normalized[1:])
    
    def test_linear_transformation(self):
        """Test that transformation is linear."""
        x1, x2 = -0.5, 0.5
        y1 = normalize_silhouette(x1)
        y2 = normalize_silhouette(x2)
        
        # Check that slope is constant (0.5 for this transformation)
        slope = (y2 - y1) / (x2 - x1)
        assert slope == pytest.approx(0.5, rel=1e-6)
    
    def test_real_scvi_biology_silhouette(self):
        """Test with real scVI biology silhouette score."""
        raw_sil = 0.434  # From notebook
        normalized = normalize_silhouette(raw_sil)
        expected = (0.434 + 1) / 2
        assert normalized == pytest.approx(expected, rel=1e-6)
        assert normalized == pytest.approx(0.717, abs=1e-3)
    
    def test_real_scvi_batch_silhouette(self):
        """Test with real scVI batch silhouette score."""
        raw_sil = 0.128  # From notebook
        normalized = normalize_silhouette(raw_sil)
        expected = (0.128 + 1) / 2
        assert normalized == pytest.approx(expected, rel=1e-6)
        assert normalized == pytest.approx(0.564, abs=1e-3)
    
    def test_preserves_variance_scaling(self):
        """Test that variance is scaled by 0.25 (linear property)."""
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        normalized = normalize_silhouette(values)
        
        var_original = np.var(values)
        var_normalized = np.var(normalized)
        
        # Linear transformation (ax + b) scales variance by a²
        # Our transformation is 0.5x + 0.5, so variance scales by 0.25
        assert var_normalized == pytest.approx(var_original * 0.25, rel=1e-6)


class TestDenormalizeSilhouette:
    """Tests for denormalize_silhouette function."""
    
    def test_inverse_perfect(self):
        """Test that denormalize is inverse of normalize for perfect score."""
        original = 1.0
        normalized = normalize_silhouette(original)
        recovered = denormalize_silhouette(normalized)
        assert recovered == pytest.approx(original, rel=1e-6)
    
    def test_inverse_worst(self):
        """Test that denormalize is inverse of normalize for worst score."""
        original = -1.0
        normalized = normalize_silhouette(original)
        recovered = denormalize_silhouette(normalized)
        assert recovered == pytest.approx(original, rel=1e-6)
    
    def test_inverse_neutral(self):
        """Test that denormalize is inverse of normalize for neutral score."""
        original = 0.0
        normalized = normalize_silhouette(original)
        recovered = denormalize_silhouette(normalized)
        assert recovered == pytest.approx(original, rel=1e-6)
    
    def test_inverse_array(self, silhouette_test_values):
        """Test inverse relationship with array input."""
        raw = silhouette_test_values['raw']
        normalized = normalize_silhouette(raw)
        recovered = denormalize_silhouette(normalized)
        np.testing.assert_allclose(recovered, raw, rtol=1e-6)
    
    def test_denormalize_range(self):
        """Test that denormalized values are in [-1, 1]."""
        normalized_values = np.linspace(0.0, 1.0, 50)
        denormalized = denormalize_silhouette(normalized_values)
        assert np.all(denormalized >= -1.0)
        assert np.all(denormalized <= 1.0)
    
    def test_specific_values(self):
        """Test specific denormalization values."""
        assert denormalize_silhouette(1.0) == pytest.approx(1.0)
        assert denormalize_silhouette(0.0) == pytest.approx(-1.0)
        assert denormalize_silhouette(0.5) == pytest.approx(0.0)
        assert denormalize_silhouette(0.75) == pytest.approx(0.5)
        assert denormalize_silhouette(0.25) == pytest.approx(-0.5)
    
    def test_round_trip_random(self):
        """Test round-trip conversion with random values."""
        np.random.seed(42)
        original = np.random.uniform(-1, 1, 100)
        normalized = normalize_silhouette(original)
        recovered = denormalize_silhouette(normalized)
        np.testing.assert_allclose(recovered, original, rtol=1e-10)


class TestNormalizationEdgeCases:
    """Test edge cases for normalization functions."""
    
    def test_normalize_boundary_values(self):
        """Test normalization at exact boundaries."""
        assert normalize_silhouette(-1.0) == 0.0
        assert normalize_silhouette(1.0) == 1.0
    
    def test_denormalize_boundary_values(self):
        """Test denormalization at exact boundaries."""
        assert denormalize_silhouette(0.0) == -1.0
        assert denormalize_silhouette(1.0) == 1.0
    
    def test_normalize_empty_array(self):
        """Test normalization with empty array."""
        result = normalize_silhouette(np.array([]))
        assert len(result) == 0
    
    def test_normalize_single_value(self):
        """Test normalization with single value."""
        result = normalize_silhouette(0.3)
        assert isinstance(result, (float, np.floating))
        assert result == pytest.approx(0.65, rel=1e-6)
    
    def test_type_preservation(self):
        """Test that input type is preserved."""
        # Float input -> float output
        result_float = normalize_silhouette(0.5)
        assert isinstance(result_float, (float, np.floating))
        
        # Array input -> array output
        result_array = normalize_silhouette(np.array([0.5]))
        assert isinstance(result_array, np.ndarray)
