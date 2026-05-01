"""Tests for model architectures."""

import pytest
import torch
import numpy as np


class TestLensClassifier:
    """Tests for LensClassifier."""

    @pytest.fixture(scope='class')
    def model(self):
        from gravlensai.models.classifier import LensClassifier
        return LensClassifier(pretrained=False, dropout=0.3)

    def test_forward_shape(self, model):
        """Output should be (B,) logits."""
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

    def test_single_image(self, model):
        """Should work with batch size 1."""
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1,)

    def test_predict_proba_range(self, model):
        """Probabilities should be in [0, 1]."""
        x = torch.randn(8, 1, 64, 64)
        probs = model.predict_proba(x)
        assert (probs >= 0).all() and (probs <= 1).all(), \
            f"Probabilities out of range: [{probs.min()}, {probs.max()}]"

    def test_extract_features_shape(self, model):
        """Feature extraction should produce (B, 512) tensor."""
        x = torch.randn(4, 1, 64, 64)
        features = model.extract_features(x)
        assert features.shape == (4, 512), f"Expected (4, 512), got {features.shape}"

    def test_gradient_flow(self, model):
        """Gradients should flow through the model."""
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients flow through ViT head (final layer)
        assert model.model.head.weight.grad is not None


class TestLensParameterRegressor:
    """Tests for LensParameterRegressor."""

    @pytest.fixture(scope='class')
    def model(self):
        from gravlensai.models.regressor import LensParameterRegressor
        return LensParameterRegressor(dropout=0.2)

    def test_forward_shape(self, model):
        """Output should be (B, 6)."""
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        assert out.shape == (4, 6), f"Expected (4, 6), got {out.shape}"

    def test_output_range(self, model):
        """Tanh output should be in [-1, 1]."""
        x = torch.randn(8, 1, 64, 64)
        out = model(x)
        assert (out >= -1).all() and (out <= 1).all(), \
            f"Output out of [-1,1]: [{out.min()}, {out.max()}]"

    def test_denormalise(self, model):
        """Denormalised outputs should be in physical ranges."""
        # Test with boundary values
        pred = torch.tensor([[-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
        phys = model.denormalise(pred)

        # Mins
        np.testing.assert_almost_equal(phys[0, 0].item(), 0.5, decimal=3)   # θ_E min
        np.testing.assert_almost_equal(phys[0, 1].item(), -0.3, decimal=3)  # e1 min
        np.testing.assert_almost_equal(phys[0, 5].item(), 8.0, decimal=3)   # M_sub min

        # Maxs
        np.testing.assert_almost_equal(phys[1, 0].item(), 2.5, decimal=3)   # θ_E max
        np.testing.assert_almost_equal(phys[1, 1].item(), 0.3, decimal=3)   # e1 max
        np.testing.assert_almost_equal(phys[1, 5].item(), 10.0, decimal=3)  # M_sub max

    def test_extract_features(self, model):
        """Feature extraction should produce (B, 512) tensor."""
        x = torch.randn(4, 1, 64, 64)
        features = model.extract_features(x)
        assert features.shape == (4, 512)


class TestDomainAdaptation:
    """Tests for domain adaptation components."""

    def test_mmd_loss_identical(self):
        """MMD loss between identical distributions should be near zero."""
        from gravlensai.models.domain_adapt import mmd_loss
        x = torch.randn(32, 512)
        loss = mmd_loss(x, x)
        assert loss.item() < 0.01, f"MMD(x, x) should be ~0, got {loss.item()}"

    def test_mmd_loss_different(self):
        """MMD loss between different distributions should be non-zero."""
        from gravlensai.models.domain_adapt import mmd_loss
        x = torch.randn(32, 512)
        y = torch.randn(32, 512) + 5.0  # shifted distribution
        loss = mmd_loss(x, y)
        assert loss.item() > 0.01, f"MMD should be non-zero for different dists"

    def test_gradient_reversal(self):
        """Gradient reversal should negate gradients."""
        from gravlensai.models.domain_adapt import GradientReversalLayer
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(4, 10, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be -1 for each element
        np.testing.assert_array_almost_equal(
            x.grad.numpy(), -np.ones_like(x.grad.numpy()), decimal=5
        )

    def test_domain_classifier_shape(self):
        """Domain classifier should output (B,) logits."""
        from gravlensai.models.domain_adapt import DomainClassifier
        dc = DomainClassifier(in_features=512, hidden=256)
        features = torch.randn(8, 512)
        out = dc(features, alpha=1.0)
        assert out.shape == (8,)

    def test_dann_alpha_schedule(self):
        """DANN alpha should increase from 0 to ~1."""
        from gravlensai.models.domain_adapt import compute_dann_alpha
        alpha_0 = compute_dann_alpha(0, 100)
        alpha_50 = compute_dann_alpha(50, 100)
        alpha_100 = compute_dann_alpha(100, 100)
        assert alpha_0 < 0.01
        assert alpha_50 > 0.4
        assert alpha_100 > 0.99


class TestUncertainty:
    """Tests for MC Dropout uncertainty quantification."""

    def test_classifier_uncertainty_shape(self):
        """MC Dropout should return correct shapes."""
        from gravlensai.models.classifier import LensClassifier
        model = LensClassifier(pretrained=False, dropout=0.3)
        x = torch.randn(2, 1, 64, 64)
        result = model.predict_with_uncertainty(x, n_forward=5)
        assert result['mean'].shape == (2,)
        assert result['std'].shape == (2,)
        assert result['predictions'].shape == (2, 5)
        assert result['entropy'].shape == (2,)

    def test_classifier_uncertainty_range(self):
        """Probabilities should be in [0,1], std >= 0."""
        from gravlensai.models.classifier import LensClassifier
        model = LensClassifier(pretrained=False, dropout=0.3)
        x = torch.randn(4, 1, 64, 64)
        result = model.predict_with_uncertainty(x, n_forward=10)
        assert (result['mean'] >= 0).all() and (result['mean'] <= 1).all()
        assert (result['std'] >= 0).all()
        assert (result['entropy'] >= 0).all()

    def test_regressor_uncertainty_shape(self):
        """Regressor MC Dropout should return correct shapes."""
        from gravlensai.models.regressor import LensParameterRegressor
        model = LensParameterRegressor(dropout=0.2)
        x = torch.randn(2, 1, 64, 64)
        result = model.predict_with_uncertainty(x, n_forward=5)
        assert result['mean'].shape == (2, 6)
        assert result['std'].shape == (2, 6)
        assert result['predictions'].shape == (2, 5, 6)
        assert result['ci_68'].shape == (2, 6, 2)
        assert result['ci_95'].shape == (2, 6, 2)

    def test_regressor_uncertainty_ci_ordering(self):
        """95% CI should be wider than 68% CI."""
        from gravlensai.models.regressor import LensParameterRegressor
        model = LensParameterRegressor(dropout=0.2)
        x = torch.randn(3, 1, 64, 64)
        result = model.predict_with_uncertainty(x, n_forward=20)
        ci68_width = result['ci_68'][:, :, 1] - result['ci_68'][:, :, 0]
        ci95_width = result['ci_95'][:, :, 1] - result['ci_95'][:, :, 0]
        assert (ci95_width >= ci68_width - 1e-5).all(), \
            "95% CI should be at least as wide as 68% CI"


class TestGradCAM:
    """Tests for Grad-CAM interpretability."""

    def test_cam_shape(self):
        """Grad-CAM output should match input spatial dims."""
        from gravlensai.models.classifier import LensClassifier
        from gravlensai.evaluate.grad_cam import GradCAM, get_target_layer
        model = LensClassifier(pretrained=False)
        target_layer = get_target_layer(model)
        cam = GradCAM(model, target_layer)
        x = torch.randn(1, 1, 64, 64)
        heatmap = cam(x)
        assert heatmap.shape == (64, 64)

    def test_cam_range(self):
        """Grad-CAM values should be normalised to [0, 1]."""
        from gravlensai.models.classifier import LensClassifier
        from gravlensai.evaluate.grad_cam import GradCAM, get_target_layer
        model = LensClassifier(pretrained=False)
        target_layer = get_target_layer(model)
        cam = GradCAM(model, target_layer)
        x = torch.randn(1, 1, 64, 64)
        heatmap = cam(x)
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0

    def test_target_layer_detection_classifier(self):
        """Should auto-detect target layer for classifier."""
        from gravlensai.models.classifier import LensClassifier
        from gravlensai.evaluate.grad_cam import get_target_layer
        model = LensClassifier(pretrained=False)
        layer = get_target_layer(model)
        assert layer is not None

    def test_target_layer_detection_regressor(self):
        """Should auto-detect target layer for regressor."""
        from gravlensai.models.regressor import LensParameterRegressor
        from gravlensai.evaluate.grad_cam import get_target_layer
        model = LensParameterRegressor()
        layer = get_target_layer(model)
        assert layer is not None


class TestEnhancedMetrics:
    """Tests for ECE, NLL, coverage metrics."""

    def test_ece_perfect_calibration(self):
        """Perfectly calibrated model should have ECE near 0."""
        from gravlensai.evaluate.metrics import expected_calibration_error
        # Create perfectly calibrated predictions
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9, 0.9])
        ece, _ = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert ece < 0.3, f"ECE should be low for well-calibrated model, got {ece}"

    def test_nll_correct_range(self):
        """NLL should be positive and finite."""
        from gravlensai.evaluate.metrics import negative_log_likelihood
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        nll = negative_log_likelihood(y_true, y_prob)
        assert nll > 0
        assert np.isfinite(nll)

    def test_nll_perfect_vs_bad(self):
        """Perfect predictions should have lower NLL than bad ones."""
        from gravlensai.evaluate.metrics import negative_log_likelihood
        y_true = np.array([0, 1, 0, 1])
        good_prob = np.array([0.01, 0.99, 0.01, 0.99])
        bad_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert negative_log_likelihood(y_true, good_prob) < \
               negative_log_likelihood(y_true, bad_prob)

    def test_coverage_probability(self):
        """Coverage should be a valid proportion."""
        from gravlensai.evaluate.metrics import coverage_probability
        y_true = np.random.randn(100, 6)
        y_pred_mean = y_true + np.random.randn(100, 6) * 0.1
        y_pred_std = np.ones((100, 6)) * 0.5
        result = coverage_probability(y_true, y_pred_mean, y_pred_std, alpha=0.68)
        for name, vals in result.items():
            assert 0 <= vals['coverage'] <= 1, f"Coverage must be in [0,1]"

