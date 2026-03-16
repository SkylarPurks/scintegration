"""
Tests for utility functions.
"""

import pytest
import numpy as np
from scintegration.utils import (
    interpret_integration_score,
    compute_theoretical_range,
    batch_effect_percentage,
    biology_preservation_percentage,
    format_score_report
)


class TestInterpretIntegrationScore:
    """Tests for interpret_integration_score function."""
    
    def test_excellent_score(self):
        """Test interpretation of excellent scores."""
        interpretation = interpret_integration_score(0.25)
        assert 'Excellent' in interpretation
        assert 'strong biology' in interpretation.lower()
    
    def test_very_good_score(self):
        """Test interpretation of very good scores."""
        interpretation = interpret_integration_score(0.15)
        assert 'Very Good' in interpretation
    
    def test_good_score(self):
        """Test interpretation of good scores."""
        interpretation = interpret_integration_score(0.07)
        assert 'Good' in interpretation
    
    def test_marginal_score(self):
        """Test interpretation of marginal scores."""
        interpretation = interpret_integration_score(0.02)
        assert 'Marginal' in interpretation
    
    def test_poor_score(self):
        """Test interpretation of poor scores."""
        interpretation = interpret_integration_score(-0.05)
        assert 'Poor' in interpretation
    
    def test_very_poor_score(self):
        """Test interpretation of very poor scores."""
        interpretation = interpret_integration_score(-0.15)
        assert 'Very Poor' in interpretation
    
    def test_zero_score(self):
        """Test interpretation of zero score."""
        interpretation = interpret_integration_score(0.0)
        assert 'Marginal' in interpretation
    
    def test_boundary_cases(self):
        """Test interpretation at score boundaries."""
        assert 'Excellent' in interpret_integration_score(0.2)
        assert 'Very Good' in interpret_integration_score(0.1)
        assert 'Good' in interpret_integration_score(0.05)
    
    def test_real_scvi_score(self):
        """Test interpretation of real scVI score."""
        interpretation = interpret_integration_score(0.0702)
        assert 'Good' in interpretation


class TestComputeTheoreticalRange:
    """Tests for compute_theoretical_range function."""
    
    def test_default_range(self):
        """Test theoretical range with default B and L ranges."""
        min_IS, max_IS = compute_theoretical_range()
        assert min_IS == pytest.approx(-0.5, rel=1e-6)
        assert max_IS == pytest.approx(0.5, rel=1e-6)
    
    def test_restricted_biology_range(self):
        """Test with restricted biology range."""
        min_IS, max_IS = compute_theoretical_range(B_range=(0.5, 1.0))
        assert min_IS >= -0.5
        assert max_IS == pytest.approx(0.5, rel=1e-6)
        assert max_IS > min_IS
    
    def test_restricted_leakage_range(self):
        """Test with restricted leakage range."""
        min_IS, max_IS = compute_theoretical_range(L_range=(0.0, 0.5))
        assert min_IS >= -0.5
        assert max_IS == pytest.approx(0.5, rel=1e-6)
    
    def test_symmetric_range(self):
        """Test that default range is symmetric around zero."""
        min_IS, max_IS = compute_theoretical_range()
        assert abs(min_IS) == pytest.approx(abs(max_IS), rel=1e-6)
    
    def test_both_restricted(self):
        """Test with both B and L restricted."""
        min_IS, max_IS = compute_theoretical_range(
            B_range=(0.6, 0.9),
            L_range=(0.3, 0.7)
        )
        assert -0.5 <= min_IS <= 0.5
        assert -0.5 <= max_IS <= 0.5
        assert min_IS < max_IS
    
    def test_edge_case_zero_ranges(self):
        """Test edge case where ranges sum to zero."""
        min_IS, max_IS = compute_theoretical_range(
            B_range=(0.0, 0.0),
            L_range=(0.0, 0.0)
        )
        assert min_IS == 0.0
        assert max_IS == 0.0


class TestBatchEffectPercentage:
    """Tests for batch_effect_percentage function."""
    
    def test_no_batch_effect(self):
        """Test with no batch effect."""
        assert batch_effect_percentage(0.0) == 0.0
    
    def test_complete_batch_effect(self):
        """Test with complete batch effect."""
        assert batch_effect_percentage(1.0) == 100.0
    
    def test_mid_range(self):
        """Test with mid-range batch effect."""
        assert batch_effect_percentage(0.5) == 50.0
    
    def test_real_scvi_batch(self):
        """Test with real scVI batch leakage."""
        pct = batch_effect_percentage(0.5597)
        assert pct == pytest.approx(55.97, abs=1e-2)
    
    def test_various_values(self):
        """Test with various L values."""
        test_cases = [
            (0.1, 10.0),
            (0.25, 25.0),
            (0.75, 75.0),
            (0.9, 90.0)
        ]
        for L, expected_pct in test_cases:
            assert batch_effect_percentage(L) == pytest.approx(expected_pct, rel=1e-6)


class TestBiologyPreservationPercentage:
    """Tests for biology_preservation_percentage function."""
    
    def test_no_biology(self):
        """Test with no biology preserved."""
        assert biology_preservation_percentage(0.0) == 0.0
    
    def test_perfect_biology(self):
        """Test with perfect biology preservation."""
        assert biology_preservation_percentage(1.0) == 100.0
    
    def test_mid_range(self):
        """Test with mid-range biology preservation."""
        assert biology_preservation_percentage(0.5) == 50.0
    
    def test_real_scvi_biology(self):
        """Test with real scVI biology score."""
        pct = biology_preservation_percentage(0.6970)
        assert pct == pytest.approx(69.70, abs=1e-2)
    
    def test_various_values(self):
        """Test with various B values."""
        test_cases = [
            (0.2, 20.0),
            (0.35, 35.0),
            (0.65, 65.0),
            (0.85, 85.0)
        ]
        for B, expected_pct in test_cases:
            assert biology_preservation_percentage(B) == pytest.approx(expected_pct, rel=1e-6)


class TestFormatScoreReport:
    """Tests for format_score_report function."""
    
    def test_basic_format(self):
        """Test basic formatting of score report."""
        report = format_score_report('scvi', 0.0702, 0.6970, 0.5597)
        
        assert isinstance(report, str)
        assert 'scvi' in report
        assert '0.0702' in report or '0.07' in report
        assert '69.70' in report or '69.7' in report
        assert '55.97' in report or '55.9' in report
    
    def test_report_contains_interpretation(self):
        """Test that report contains interpretation."""
        report = format_score_report('test', 0.15, 0.8, 0.5)
        assert 'Very Good' in report or 'Good' in report
    
    def test_report_structure(self):
        """Test that report has proper structure."""
        report = format_score_report('scvi', 0.0702, 0.6970, 0.5597)
        
        lines = report.split('\n')
        assert len(lines) >= 5  # At least model name, separator, score, bio, batch
        assert 'MODEL:' in lines[0] or 'MODEL' in lines[0]
    
    def test_negative_score(self):
        """Test formatting with negative integration score."""
        report = format_score_report('pca', -0.0303, 0.6332, 0.6867)
        
        assert '-0.03' in report or '-0.030' in report
        assert '63.32' in report or '63.3' in report
        assert '68.67' in report or '68.6' in report
    
    def test_multiple_models(self):
        """Test formatting reports for multiple models."""
        models = [
            ('scvi', 0.0702, 0.6970, 0.5597),
            ('scgpt', 0.0533, 0.6822, 0.5764),
            ('pca', -0.0303, 0.6332, 0.6867)
        ]
        
        reports = [format_score_report(*model) for model in models]
        
        assert len(reports) == 3
        for i, (model_name, _, _, _) in enumerate(models):
            assert model_name in reports[i]
    
    def test_extreme_values(self):
        """Test formatting with extreme values."""
        # Perfect integration
        report1 = format_score_report('perfect', 0.5, 1.0, 0.0)
        assert '0.5' in report1 or '0.50' in report1
        assert '100' in report1
        
        # Worst integration
        report2 = format_score_report('worst', -0.5, 0.0, 1.0)
        assert '-0.5' in report2 or '-0.50' in report2


class TestUtilityEdgeCases:
    """Test edge cases for utility functions."""
    
    def test_interpret_out_of_range_high(self):
        """Test interpretation of score above theoretical max."""
        # Should still work even if > 0.5
        interpretation = interpret_integration_score(0.6)
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
    
    def test_interpret_out_of_range_low(self):
        """Test interpretation of score below theoretical min."""
        # Should still work even if < -0.5
        interpretation = interpret_integration_score(-0.6)
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
    
    def test_percentages_with_floats(self):
        """Test percentage functions handle floats correctly."""
        # Test with various float precisions
        assert batch_effect_percentage(0.123456) == pytest.approx(12.3456, rel=1e-6)
        assert biology_preservation_percentage(0.987654) == pytest.approx(98.7654, rel=1e-6)
    
    def test_format_score_with_zero_scores(self):
        """Test formatting when scores are zero."""
        report = format_score_report('zero', 0.0, 0.0, 0.0)
        assert '0.0' in report or '0.00' in report
        assert 'zero' in report.lower()
