"""
Vision Transformer (ViT) based gravitational lens classifier.

Architecture choices:
- timm 'vit_tiny_patch16_224' dynamically resized for 64x64 images using 8x8 patches.
- 1 input channel (grayscale).
- Final linear layer outputs 1 logit (binary BCE loss).
- Sigmoid applied at inference time for probability.

Reference baseline: Jacobs et al. 2017 achieved 90%+ precision on simulated data
with CNNs, but ViTs capture global spatial dependencies for subhalo detection.
"""

import torch
import torch.nn as nn
import timm


class LensClassifier(nn.Module):
    """
    Binary classifier: lensed (1) vs. non-lensed (0).

    Input:  (B, 1, 64, 64) float32 tensor, arcsinh-normalised
    Output: (B,) raw logit (apply sigmoid for probability)
    """

    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()

        # We use a tiny Vision Transformer, adjusting the patch size for 64x64 input
        # 64 / 8 = 8 patches per side, resulting in an 8x8 grid = 64 tokens + CLS token.
        self.model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            img_size=64,
            patch_size=8,
            in_chans=1,
            num_classes=1,
            drop_rate=dropout,
            attn_drop_rate=dropout
        )
        self.feature_projection = nn.Linear(self.model.num_features, 512)
        self.features = nn.Sequential(
            self.model.patch_embed.proj,
            nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 1, 64, 64) input tensor.

        Returns:
            (B,) raw logits. Apply sigmoid for probabilities.
        """
        return self.model(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for inference."""
        return torch.sigmoid(self.forward(x))

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_forward: int = 30
    ) -> dict:
        """
        MC Dropout uncertainty estimation.

        Performs N stochastic forward passes with dropout enabled at inference
        time. The variance across passes quantifies epistemic uncertainty —
        how unsure the model is due to limited training data.

        Args:
            x: (B, 1, 64, 64) input tensor.
            n_forward: Number of stochastic passes (default 30).

        Returns:
            Dict with keys:
                mean: (B,) mean predicted probability.
                std: (B,) standard deviation (epistemic uncertainty).
                predictions: (B, N) all individual predictions.
                entropy: (B,) predictive entropy in bits.
        """
        # Enable ONLY dropout layers for MC Dropout (keep BatchNorm frozen!)
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        preds = torch.stack(
            [torch.sigmoid(self.forward(x)) for _ in range(n_forward)], dim=1
        )
        self.eval()

        mean = preds.mean(dim=1)
        std = preds.std(dim=1)

        # Predictive entropy (Shannon, in bits)
        eps = 1e-8
        entropy = -(
            mean * torch.log2(mean + eps)
            + (1 - mean) * torch.log2(1 - mean + eps)
        )

        return {
            'mean': mean,
            'std': std,
            'predictions': preds,
            'entropy': entropy,
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings from the ViT backbone (for domain adaptation).

        Returns:
            (B, 512) projected CLS token feature tensor from the ViT backbone.
        """
        # ViT backbone uses forward_features; the CLS token is at index 0
        features = self.model.forward_features(x)  # (B, N_tokens, embed_dim)
        cls_features = features[:, 0, :]
        return self.feature_projection(cls_features)
