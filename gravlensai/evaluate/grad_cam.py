"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for GravLensAI.

Generates heatmaps showing which spatial regions of an input image most
influence the model's prediction. For gravitational lens detection, we
expect the model to attend to Einstein ring/arc structures.

Reference: Selvaraju et al. 2017, "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization", ICCV.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class GradCAM:
    """
    Grad-CAM visualisation for CNN models.

    Hooks into a target convolutional layer, captures the activations
    and gradients during a forward + backward pass, then computes a
    class-weighted spatial heatmap.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-3])
        heatmap = cam(image_tensor)
        cam.overlay(image_np, heatmap)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for an input image.

        Args:
            x: (1, 1, H, W) or (1, C, H, W) input tensor.
            target_class: Class index for gradient computation.
                          For binary classifiers, use None (default).

        Returns:
            (H, W) numpy array, normalised to [0, 1].
        """
        self.model.eval()
        x = x.requires_grad_(True)

        # Forward pass
        output = self.model(x)

        # For binary classifier: single logit → use directly
        if output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
            score = output.squeeze()
        else:
            # Multi-class: use target_class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            score = output[0, target_class]

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")

        # Grad-CAM computation
        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Only positive contributions

        # Resize to input spatial dimensions
        cam = F.interpolate(
            cam, size=x.shape[2:], mode='bilinear', align_corners=False
        )

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def generate_cam_for_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmaps for a batch of images.

    Args:
        model: Trained CNN model.
        images: (B, 1, H, W) input batch.
        target_layer: Target convolutional layer for Grad-CAM.

    Returns:
        (B, H, W) numpy array of heatmaps.
    """
    cam_extractor = GradCAM(model, target_layer)
    heatmaps = []

    for i in range(images.shape[0]):
        img = images[i:i+1]  # Keep batch dim
        heatmap = cam_extractor(img)
        heatmaps.append(heatmap)

    return np.stack(heatmaps)


def plot_cam_grid(
    images: np.ndarray,
    heatmaps: np.ndarray,
    probs: Optional[np.ndarray] = None,
    n: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Publication-quality Grad-CAM visualisation grid.

    Shows original image, heatmap overlay, and pure heatmap side-by-side
    for the top-N most confident lens detections.

    Args:
        images: (N, H, W) input images.
        heatmaps: (N, H, W) Grad-CAM heatmaps.
        probs: (N,) predicted probabilities (for sorting).
        n: Number of examples to show.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    if probs is not None:
        idx = np.argsort(-probs)[:n]
    else:
        idx = np.arange(min(n, len(images)))

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    fig.patch.set_facecolor('#0a0a0a')

    col_titles = ['Input Image', 'Grad-CAM Overlay', 'Attention Heatmap']

    for row, data_i in enumerate(idx):
        img = images[data_i]
        cam = heatmaps[data_i]

        vmin, vmax = np.percentile(img, [1, 99])

        # Column 1: Original image
        ax = axes[row, 0]
        ax.imshow(img, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        if probs is not None:
            ax.set_ylabel(f'p={probs[data_i]:.3f}', color='white',
                         fontsize=10, rotation=0, labelpad=50)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 2: Overlay
        ax = axes[row, 1]
        ax.imshow(img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.imshow(cam, cmap='jet', origin='lower', alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 3: Pure heatmap
        ax = axes[row, 2]
        im = ax.imshow(cam, cmap='magma', origin='lower', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    # Column headers
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, color='white', fontsize=11, pad=8)

    fig.suptitle(
        'Grad-CAM — Where Does GravLensAI Look?',
        color='white', fontsize=14, y=1.01
    )
    plt.tight_layout()

    if save_path:
        from pathlib import Path as P
        P(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='#0a0a0a')

    return fig


def get_target_layer(model) -> torch.nn.Module:
    """
    Automatically select the appropriate target layer for Grad-CAM.

    For ResNet-18 (classifier): uses the last conv block (layer4).
    For the custom CNN (regressor): uses the last conv layer before avgpool.

    Args:
        model: A LensClassifier or LensParameterRegressor instance.

    Returns:
        Target convolutional layer.
    """
    # ResNet-18 classifier — layer4 is the last residual block
    if hasattr(model, 'features') and hasattr(model.features, '__getitem__'):
        children = list(model.features.children())
        # Walk backwards to find last Conv2d or Sequential block
        for layer in reversed(children):
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Sequential)):
                return layer
    # Fallback: last named module that is Conv2d
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv
    raise ValueError("Could not find target convolutional layer for Grad-CAM")
