"""
Tests for core integration score calculation functions.
"""

import pytest
import numpy as np
from scintegration.core import (
    calculate_integration_score,
    compute_B_score,
    compute_L_score
)


class TestCalculateIntegrationScore:
    """Tests for calculate_integration_score function."""
    
    def test_positive_score(self):
        """Test when biology > batch (positive score)."""
        B, L = 0.7, 0.5
        score = calculate_integration_score(B, L)
        expected = (0.7 - 0.5) / (2 * (0.7 + 0.5))
        assert score == pytest.approx(expected, rel=1e-6)
        assert score > 0
    
    def test_negative_score(self):
        """Test when batch > biology (negative score)."""
        B, L = 0.3, 0.7
        score = calculate_integration_score(B, L)
        expected = (0.3 - 0.7) / (2 * (0.3 + 0.7))
        assert score == pytest.approx(expected, rel=1e-6)
        assert score < 0
    
    def test_equal_scores(self):
        """Test when biology equals batch (zero score)."""
        score = calculate_integration_score(0.5, 0.5)
        assert score == 0.0
    
    def test_zero_sum_edge_case(self):
        """Test edge case when both B and L are zero."""
        score = calculate_integration_score(0.0, 0.0)
        assert score == 0.0
    
    def test_perfect_integration(self):
        """Test perfect integration: high biology, no batch."""
        score = calculate_integration_score(1.0, 0.0)
        assert score == 0.5
    
    def test_worst_integration(self):
        """Test worst integration: no biology, high batch."""
        score = calculate_integration_score(0.0, 1.0)
        assert score == -0.5
    
    def test_score_range(self):
        """Test that score is always in [-0.5, 0.5] range."""
        test_cases = [
            (0.1, 0.9), (0.9, 0.1),
            (0.3, 0.7), (0.7, 0.3),
            (0.5, 0.5), (1.0, 1.0)
        ]
        for B, L in test_cases:
            score = calculate_integration_score(B, L)
            assert -0.5 <= score <= 0.5, f"Score {score} out of range for B={B}, L={L}"
    
    def test_real_data_scvi(self, expected_integration_scores):
        """Test with real scVI data from notebook."""
        expected = expected_integration_scores['scvi']
        score = calculate_integration_score(expected['B'], expected['L'])
        assert score == pytest.approx(expected['IntegrationScore'], abs=1e-4)
    
    def test_real_data_scgpt(self, expected_integration_scores):
        """Test with real scGPT data from notebook."""
        expected = expected_integration_scores['scgpt']
        score = calculate_integration_score(expected['B'], expected['L'])
        assert score == pytest.approx(expected['IntegrationScore'], abs=1e-4)
    
    def test_real_data_pca(self, expected_integration_scores):
        """Test with real PCA data from notebook."""
        expected = expected_integration_scores['pca']
        score = calculate_integration_score(expected['B'], expected['L'])
        assert score == pytest.approx(expected['IntegrationScore'], abs=1e-4)
    
    def test_monotonicity_in_B(self):
        """Test that score increases when B increases (L fixed)."""
        L = 0.5
        scores = [calculate_integration_score(B, L) for B in [0.2, 0.4, 0.6, 0.8]]
        assert all(scores[i] < scores[i+1] for i in range(len(scores)-1))
    
    def test_monotonicity_in_L(self):
        """Test that score decreases when L increases (B fixed)."""
        B = 0.7
        scores = [calculate_integration_score(B, L) for L in [0.2, 0.4, 0.6, 0.8]]
        assert all(scores[i] > scores[i+1] for i in range(len(scores)-1))


class TestComputeBScore:
    """Tests for compute_B_score function."""
    
    def test_equal_weights_average(self):
        """Test that equal weights produces simple average."""
        score = compute_B_score(0.8, 0.6, 0.7, 0.5, weights='equal')
        expected = (0.8 + 0.6 + 0.7 + 0.5) / 4
        assert score == pytest.approx(expected, rel=1e-6)
    
    def test_all_perfect_scores(self):
        """Test with all metrics at 1.0."""
        score = compute_B_score(1.0, 1.0, 1.0, 1.0)
        assert score == 1.0
    
    def test_all_zero_scores(self):
        """Test with all metrics at 0.0."""
        score = compute_B_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0.0
    
    def test_custom_weights(self):
        """Test with custom weights."""
        weights = {'ari': 0.3, 'nmi': 0.3, 'sil': 0.2, 'f1': 0.2}
        score = compute_B_score(0.7, 0.8, 0.6, 0.5, weights=weights)
        expected = 0.3*0.7 + 0.3*0.8 + 0.2*0.6 + 0.2*0.5
        assert score == pytest.approx(expected, rel=1e-6)
    
    def test_real_scvi_biology(self):
        """Test with real scVI biology metrics."""
        # ARI=0.732, NMI=0.785, Sil_normalized=0.717, F1=0.554
        score = compute_B_score(0.732, 0.785, 0.717, 0.554)
        expected = (0.732 + 0.785 + 0.717 + 0.554) / 4
        assert score == pytest.approx(expected, abs=1e-4)
        assert score == pytest.approx(0.6970, abs=1e-4)
    
    def test_output_range(self):
        """Test that output is in [0, 1] range."""
        test_cases = [
            (0.1, 0.2, 0.3, 0.4),
            (0.9, 0.8, 0.7, 0.6),
            (0.5, 0.5, 0.5, 0.5),
        ]
        for ari, nmi, sil, f1 in test_cases:
            score = compute_B_score(ari, nmi, sil, f1)
            assert 0.0 <= score <= 1.0


class TestComputeLScore:
    """Tests for compute_L_score function."""
    
    def test_equal_weights_average(self):
        """Test that equal weights produces simple average."""
        score = compute_L_score(0.6, 0.5, 0.4, 0.3, weights='equal')
        expected = (0.6 + 0.5 + 0.4 + 0.3) / 4
        assert score == pytest.approx(expected, rel=1e-6)
    
    def test_perfect_batch_removal(self):
        """Test perfect batch removal (all zeros)."""
        score = compute_L_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0.0
    
    def test_complete_batch_leakage(self):
        """Test complete batch leakage (all ones)."""
        score = compute_L_score(1.0, 1.0, 1.0, 1.0)
        assert score == 1.0
    
    def test_custom_weights(self):
        """Test with custom weights."""
        weights = {'ari': 0.4, 'nmi': 0.3, 'sil': 0.2, 'f1': 0.1}
        score = compute_L_score(0.6, 0.5, 0.4, 0.3, weights=weights)
        expected = 0.4*0.6 + 0.3*0.5 + 0.2*0.4 + 0.1*0.3
        assert score == pytest.approx(expected, rel=1e-6)
    
    def test_real_scvi_batch(self):
        """Test with real scVI batch metrics."""
        # ARI=0.605, NMI=0.657, Sil_normalized=0.564, F1=0.413
        score = compute_L_score(0.605, 0.657, 0.564, 0.413)
        expected = (0.605 + 0.657 + 0.564 + 0.413) / 4
        assert score == pytest.approx(expected, abs=1e-4)
        assert score == pytest.approx(0.5597, abs=1e-4)
    
    def test_same_logic_as_B_score(self):
        """Test that L and B scores use same averaging logic."""
        values = (0.7, 0.6, 0.5, 0.4)
        b_score = compute_B_score(*values)
        l_score = compute_L_score(*values)
        assert b_score == l_score
    
    def test_output_range(self):
        """Test that output is in [0, 1] range."""
        test_cases = [
            (0.1, 0.2, 0.3, 0.4),
            (0.9, 0.8, 0.7, 0.6),
            (0.5, 0.5, 0.5, 0.5),
        ]
        for ari, nmi, sil, f1 in test_cases:
            score = compute_L_score(ari, nmi, sil, f1)
            assert 0.0 <= score <= 1.0


class TestIntegrationScoreProperties:
    """Test mathematical properties of the integration score."""
    
    def test_symmetry(self):
        """Test that swapping B and L negates the score."""
        B, L = 0.7, 0.3
        score1 = calculate_integration_score(B, L)
        score2 = calculate_integration_score(L, B)
        assert score1 == pytest.approx(-score2, rel=1e-6)
    
    def test_scale_invariance(self):
        """Test that scaling both B and L equally doesn't change score."""
        B, L = 0.6, 0.4
        score1 = calculate_integration_score(B, L)
        score2 = calculate_integration_score(B*0.5, L*0.5)
        assert score1 == pytest.approx(score2, rel=1e-6)
    
    def test_continuous(self):
        """Test that score changes continuously with inputs."""
        B = 0.7
        L_values = np.linspace(0.1, 0.9, 100)
        scores = [calculate_integration_score(B, L) for L in L_values]
        # Check that adjacent scores are close
        diffs = np.abs(np.diff(scores))
        assert np.all(diffs < 0.05)  # No sudden jumps
