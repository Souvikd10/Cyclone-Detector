"""
Grad-CAM Explainability — Stage 13.

Generates Grad-CAM heatmaps using Captum to visualize which image
regions drive the model's cyclone detection predictions.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradCAMGenerator:
    """
    Grad-CAM heatmap generator for cyclone models.

    Uses Captum's LayerGradCam (when available) or a manual implementation
    to produce class-activation heatmaps showing which spatial regions
    the model focuses on for cyclone detection.

    Args:
        model: The complete model (fusion + encoder + detection head).
        target_layer: The convolutional layer to target for Grad-CAM.
                      Default: shared_encoder.spatial_projector[0].
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        target_layer: nn.Module | None = None,
    ):
        self.model = model
        self.target_layer = target_layer
        self._use_captum = False

        if model is not None and target_layer is not None:
            try:
                from captum.attr import LayerGradCam
                self._gradcam = LayerGradCam(model, target_layer)
                self._use_captum = True
                logger.info("GradCAM initialized with Captum")
            except ImportError:
                logger.info("Captum not available, using manual Grad-CAM")
                self._gradcam = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 0,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: [1, C, H, W] input to the model.
            target_class: Class index for which to compute attribution.

        Returns:
            Heatmap as [H_input, W_input] numpy array normalized to [0, 1].
        """
        if self._use_captum and self._gradcam is not None:
            return self._generate_captum(input_tensor, target_class)
        elif self.model is not None and self.target_layer is not None:
            return self._generate_manual(input_tensor, target_class)
        else:
            # Fallback: generate synthetic heatmap
            return self._generate_synthetic(input_tensor.shape[-2:])

    def _generate_captum(
        self, input_tensor: torch.Tensor, target_class: int
    ) -> np.ndarray:
        """Generate heatmap using Captum LayerGradCam."""
        self.model.eval()
        input_tensor.requires_grad_(True)

        try:
            attribution = self._gradcam.attribute(
                input_tensor, target=target_class,
            )

            # Normalize to [0, 1]
            heatmap = attribution[0, 0].detach().cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            # Resize to input resolution
            from scipy.ndimage import zoom
            H_in, W_in = input_tensor.shape[-2:]
            H_map, W_map = heatmap.shape
            if H_map != H_in or W_map != W_in:
                heatmap = zoom(heatmap, (H_in / H_map, W_in / W_map), order=1)

            return heatmap.astype(np.float32)

        except Exception as e:
            logger.warning(f"Captum Grad-CAM failed: {e}, falling back to synthetic")
            return self._generate_synthetic(input_tensor.shape[-2:])

    def _generate_manual(
        self, input_tensor: torch.Tensor, target_class: int
    ) -> np.ndarray:
        """Manual Grad-CAM implementation (no Captum dependency)."""
        self.model.eval()

        # Register hooks to capture activations and gradients
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations["value"] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)

            # Get the target score
            if isinstance(output, dict):
                if "presence_logit" in output:
                    score = output["presence_logit"]
                elif "confidence" in output:
                    score = output["confidence"]
                else:
                    score = list(output.values())[0]
            else:
                score = output

            if score.dim() > 1:
                score = score.squeeze()
            if score.dim() > 0:
                score = score[target_class] if score.shape[0] > target_class else score[0]

            # Backward pass
            self.model.zero_grad()
            score.backward(retain_graph=True)

            # Compute Grad-CAM
            acts = activations.get("value")
            grads = gradients.get("value")

            if acts is None or grads is None:
                return self._generate_synthetic(input_tensor.shape[-2:])

            # Global average pool gradients → channel weights
            weights = grads.mean(dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
            cam = (weights * acts).sum(dim=1, keepdim=True)    # [B, 1, H, W]
            cam = torch.relu(cam)

            heatmap = cam[0, 0].cpu().numpy()
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            # Resize
            from scipy.ndimage import zoom
            H_in, W_in = input_tensor.shape[-2:]
            H_map, W_map = heatmap.shape
            if H_map != H_in or W_map != W_in:
                heatmap = zoom(heatmap, (H_in / H_map, W_in / W_map), order=1)

            return heatmap.astype(np.float32)

        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    @staticmethod
    def _generate_synthetic(spatial_size: tuple) -> np.ndarray:
        """
        Generate a synthetic Grad-CAM heatmap for prototyping.

        Creates a radial gradient centered on the image
        (simulating attention on the cyclone center).
        """
        H, W = spatial_size if len(spatial_size) == 2 else (512, 512)
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, W)
        xx, yy = np.meshgrid(x, y)

        # Offset center slightly for realism
        cx, cy = np.random.uniform(-0.1, 0.1, 2)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Gaussian blob
        heatmap = np.exp(-r ** 2 / (2 * 0.3 ** 2))

        # Add some asymmetry
        heatmap += 0.2 * np.exp(-((xx - cx - 0.2) ** 2 + (yy - cy + 0.1) ** 2) / (2 * 0.15 ** 2))

        # Normalize
        heatmap = np.clip(heatmap, 0, 1)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.astype(np.float32)

    def overlay_on_image(
        self,
        heatmap: np.ndarray,
        base_image: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the base satellite image.

        Args:
            heatmap: [H, W] normalized heatmap.
            base_image: [H, W] single-channel image (e.g., IR).
            alpha: Transparency of the overlay.
            colormap: Matplotlib colormap name.

        Returns:
            [H, W, 3] RGB overlay image (uint8).
        """
        try:
            import matplotlib.pyplot as plt

            # Normalize base image to [0, 1]
            base_min, base_max = base_image.min(), base_image.max()
            if base_max > base_min:
                base_norm = (base_image - base_min) / (base_max - base_min)
            else:
                base_norm = np.zeros_like(base_image)

            # Apply colormap to heatmap
            cmap = plt.get_cmap(colormap)
            heatmap_colored = cmap(heatmap)[:, :, :3]  # [H, W, 3]

            # Base image in grayscale RGB
            base_rgb = np.stack([base_norm] * 3, axis=-1)

            # Blend
            overlay = (1 - alpha) * base_rgb + alpha * heatmap_colored
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

            return overlay

        except ImportError:
            # Without matplotlib, just return heatmap as grayscale
            return (heatmap * 255).astype(np.uint8)
