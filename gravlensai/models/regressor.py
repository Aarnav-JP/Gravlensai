"""
CNN regression network estimating lens parameters from an image.

Follows the architecture of Hezaveh et al. 2017 (Nature):
- ResNet-50 feature extractor with BatchNorm
- Outputs continuous values normalised to [-1, 1] via Tanh
- At inference, denormalise to recover physical units

The default contract is 6 parameters, including the subhalo mass target, but
older 5-parameter checkpoints remain supported by setting ``output_dim=5``.

Parameters (after denorm, default 6-target mode):
    θ_E: Einstein radius [0.5, 2.5] arcsec
    e1:  ellipticity component 1 [-0.3, 0.3]
    e2:  ellipticity component 2 [-0.3, 0.3]
    γ1:  external shear 1 [-0.05, 0.05]
    γ2:  external shear 2 [-0.05, 0.05]
    M_sub: Dark Matter subhalo mass (Log10 M_sun) [8.0, 10.0]
"""

import torch
import torch.nn as nn
import torchvision.models as models


class LensParameterRegressor(nn.Module):
    """
    ResNet-50 regression network estimating lens parameters from an image.

    Input:  (B, 1, 64, 64)
    Output: (B, N) — normalised [-1, 1] parameters, where N defaults to 6.
    """

    # Physical parameter ranges for denormalisation
    PARAM_MINS = torch.tensor([0.5, -0.3, -0.3, -0.05, -0.05, 8.0])
    PARAM_MAXS = torch.tensor([2.5,  0.3,  0.3,  0.05,  0.05, 10.0])

    def __init__(self, dropout: float = 0.2, output_dim: int = 6):
        super().__init__()
        if output_dim < 1 or output_dim > len(self.PARAM_MINS):
            raise ValueError(
                f"output_dim must be between 1 and {len(self.PARAM_MINS)}, got {output_dim}"
            )
        self.output_dim = output_dim

        # Load ResNet-50 (no pre-trained weights since astronomical data is very different from ImageNet)
        base_model = models.resnet50(weights=None)

        # Modify the first convolutional layer to accept 1-channel (grayscale) inputs instead of 3
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Feature extraction: use all ResNet layers up to the adaptive pooling layer
        # This allows extract_features() to still work seamlessly.
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
            nn.Flatten()
        )

        # Regression head with Dropout for Monte Carlo (MC) uncertainty estimation
        # ResNet-50's fc layer has 2048 input features.
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh(),  # Output in [-1, 1] matching normalised targets
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 1, 64, 64) input tensor.

        Returns:
            (B, N) predictions in normalised [-1, 1] space.
        """
        return self.regressor(self.features(x))

    def denormalise(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Convert normalised [-1,1] predictions back to physical units.

        Args:
            pred: (B, N) normalised predictions.

        Returns:
            (B, N) predictions in physical units.
        """
        mins = self.PARAM_MINS[: pred.shape[1]].to(pred.device)
        maxs = self.PARAM_MAXS[: pred.shape[1]].to(pred.device)
        return (pred + 1) / 2 * (maxs - mins) + mins

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_forward: int = 30
    ) -> dict:
        """
        MC Dropout uncertainty estimation for parameter regression.

        Performs N stochastic forward passes with dropout enabled at inference
        time. Returns per-parameter mean and standard deviation in physical
        units, enabling error bars (e.g., θ_E = 1.2 ± 0.08 arcsec).

        Args:
            x: (B, 1, 64, 64) input tensor.
            n_forward: Number of stochastic passes (default 30).

        Returns:
            Dict with keys:
                mean: (B, N) mean predicted parameters in physical units.
                std: (B, N) per-parameter standard deviation (uncertainty).
                predictions: (B, n_forward, N) all individual predictions.
                ci_68: (B, N, 2) 68% confidence interval [lower, upper].
                ci_95: (B, N, 2) 95% confidence interval [lower, upper].
        """
        # Enable ONLY dropout layers for MC Dropout (keep BatchNorm frozen!)
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        raw_preds = torch.stack(
            [self.forward(x) for _ in range(n_forward)], dim=1
        )  # (B, N, 5) normalised
        self.eval()

        # Denormalise all predictions to physical units
        batch_size, num_forward, num_params = raw_preds.shape
        phys_preds = self.denormalise(raw_preds.view(batch_size * num_forward, num_params)).view(batch_size, num_forward, num_params)

        mean = phys_preds.mean(dim=1)  # (B, 5)
        std = phys_preds.std(dim=1)    # (B, 5)

        # Confidence intervals from quantiles
        ci_68_lo = phys_preds.quantile(0.16, dim=1)  # (B, 5)
        ci_68_hi = phys_preds.quantile(0.84, dim=1)
        ci_95_lo = phys_preds.quantile(0.025, dim=1)
        ci_95_hi = phys_preds.quantile(0.975, dim=1)

        return {
            'mean': mean,
            'std': std,
            'predictions': phys_preds,
            'ci_68': torch.stack([ci_68_lo, ci_68_hi], dim=-1),
            'ci_95': torch.stack([ci_95_lo, ci_95_hi], dim=-1),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (for analysis or domain adaptation).

        Returns:
            (B, 512) feature tensor.
        """
        features = self.features(x)
        return features.view(features.size(0), -1)
