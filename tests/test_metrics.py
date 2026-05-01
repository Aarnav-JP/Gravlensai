"""Tests for evaluation metrics."""

import pytest
import numpy as np

from gravlensai.evaluate.metrics import (
    classifier_metrics, optimal_threshold,
    regressor_metrics, speedup_vs_lenstool,
)


class TestClassifierMetrics:
    """Tests for classifier_metrics."""

    def test_perfect_classifier(self):
        """Perfect predictions should give precision=recall=1."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.9, 0.95, 1.0])
        m = classifier_metrics(y_true, y_prob)
        assert m['precision'] == 1.0
        assert m['recall'] == 1.0
        assert m['f1'] == 1.0
        assert m['auc_roc'] > 0.99

    def test_worst_classifier(self):
        """Completely wrong predictions should give precision=0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = classifier_metrics(y_true, y_prob)
        assert m['precision'] == 0.0
        assert m['recall'] == 0.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2×2."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        m = classifier_metrics(y_true, y_prob)
        cm = m['confusion_matrix']
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_custom_threshold(self):
        """Different threshold should change predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 0.6, 0.7, 0.8])

        m_05 = classifier_metrics(y_true, y_prob, threshold=0.5)
        m_09 = classifier_metrics(y_true, y_prob, threshold=0.9)
        # Higher threshold → fewer positives → higher precision or lower recall
        assert m_09['recall'] <= m_05['recall']


class TestOptimalThreshold:
    """Tests for optimal_threshold."""

    def test_returns_float(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        t = optimal_threshold(y_true, y_prob)
        assert isinstance(t, float)
        assert 0.0 <= t <= 1.0


class TestRegressorMetrics:
    """Tests for regressor_metrics."""

    def test_zero_error(self):
        """Identical pred/true should give 0 RMSE."""
        y = np.random.randn(50, 6).astype(np.float32)
        m = regressor_metrics(y, y)
        for name, vals in m.items():
            if isinstance(vals, dict):
                assert vals['rmse'] < 1e-6, f"{name} RMSE should be ~0"
                assert vals['median'] < 1e-6

    def test_known_error(self):
        """Known constant offset should give predictable RMSE."""
        y_true = np.zeros((100, 6), dtype=np.float32)
        y_pred = np.ones((100, 6), dtype=np.float32) * 0.5
        m = regressor_metrics(y_true, y_pred)
        for name, vals in m.items():
            if isinstance(vals, dict):
                np.testing.assert_almost_equal(vals['rmse'], 0.5, decimal=3)

    def test_per_parameter_names(self):
        """Should return metrics for all 6 parameters."""
        y = np.random.randn(20, 6).astype(np.float32)
        m = regressor_metrics(y, y)
        expected_names = ['Einstein_radius_arcsec', 'e1', 'e2', 'gamma1', 'gamma2', 'subhalo_mass']
        for name in expected_names:
            assert name in m, f"Missing parameter: {name}"


class TestSpeedupMetrics:
    """Tests for speedup_vs_lenstool."""

    def test_speedup_positive(self):
        speed = speedup_vs_lenstool(100, 0.5)
        assert speed['speedup_factor'] > 1000
        assert speed['cnn_ms_per_image'] > 0

    def test_speedup_values(self):
        # 100 images in 1 second = 10ms each
        # LENSTOOL default = 120s each
        # Speedup = 120/0.01 = 12000x
        speed = speedup_vs_lenstool(100, 1.0)
        np.testing.assert_almost_equal(
            speed['cnn_ms_per_image'], 10.0, decimal=1
        )
        np.testing.assert_almost_equal(
            speed['speedup_factor'], 12000.0, decimal=0
        )
