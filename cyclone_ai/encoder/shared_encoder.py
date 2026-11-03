"""
Shared Multi-Source Encoder — Stage 4.

Takes fused feature maps from Stage 3 and produces rich storm features
using a pretrained backbone (ConvNeXt-Tiny or ViT-Small via timm).

Outputs:
- Global feature vector [B, feature_dim] for classification/RI heads
- Spatial feature map [B, feature_dim, H', W'] for detection/structure heads
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SharedEncoder(nn.Module):
    """
    Shared multi-source encoder using a pretrained backbone.

    Takes the fused feature map from the AttentionGatedFusion module
    and extracts rich storm features. The backbone is selected via config.

    Supported backbones:
    - 'convnext_tiny': ConvNeXt-Tiny (default, good for Grad-CAM)
    - 'convnext_small': ConvNeXt-Small (more capacity)
    - 'vit_small_patch16_224': ViT-Small (global attention, needs resize to 224)

    Args:
        backbone_name: timm model name.
        pretrained: Whether to load ImageNet pretrained weights.
        input_channels: Number of input channels (from fusion output).
        feature_dim: Output feature vector dimension.
        drop_path_rate: Drop path rate for backbone.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        input_channels: int = 256,
        feature_dim: int = 768,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.feature_dim = feature_dim

        # Input adapter: map fused channels to backbone's expected input
        # timm models expect 3-channel (RGB) input by default
        self.input_adapter = nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )

        # Load backbone from timm
        try:
            import timm

            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,       # Get intermediate feature maps
                drop_path_rate=drop_path_rate,
            )

            # Get the output channels of the last feature level
            # Use a dummy forward pass to determine dimensions
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                features = self.backbone(dummy)
                last_feat = features[-1]
                backbone_out_channels = last_feat.shape[1]
                self._spatial_scale = 224 // last_feat.shape[2]

            logger.info(
                f"Loaded backbone '{backbone_name}': "
                f"output channels={backbone_out_channels}, "
                f"spatial scale=1/{self._spatial_scale}"
            )
        except ImportError:
            logger.warning("timm not available, using fallback simple backbone")
            self.backbone = None
            backbone_out_channels = 256
            self._spatial_scale = 16

        # Project backbone output to desired feature_dim
        self.spatial_projector = nn.Sequential(
            nn.Conv2d(backbone_out_channels, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

        # Global feature extractor (average pool + projection)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
        )

        # Fallback simple backbone if timm not available
        if self.backbone is None:
            self._fallback_backbone = self._build_fallback(3, backbone_out_channels)

    def _build_fallback(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Simple CNN backbone fallback when timm is not installed."""
        return nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extract storm features from fused input.

        Args:
            x: [B, input_channels, H, W] fused feature map from Stage 3.

        Returns:
            Dict with:
                'spatial': [B, feature_dim, H', W'] spatial feature map
                'global': [B, feature_dim] global feature vector
                'backbone_features': list of intermediate feature maps (for Grad-CAM)
        """
        # Adapt input channels
        x = self.input_adapter(x)

        # Extract features via backbone
        if self.backbone is not None:
            backbone_features = self.backbone(x)  # list of feature maps
            last_feat = backbone_features[-1]      # [B, C_backbone, H', W']
        else:
            last_feat = self._fallback_backbone(x)
            backbone_features = [last_feat]

        # Project spatial features
        spatial = self.spatial_projector(last_feat)  # [B, feature_dim, H', W']

        # Global feature
        pooled = self.global_pool(spatial).squeeze(-1).squeeze(-1)  # [B, feature_dim]
        global_feat = self.global_projector(pooled)  # [B, feature_dim]

        return {
            "spatial": spatial,
            "global": global_feat,
            "backbone_features": backbone_features,
        }

    def get_last_conv_layer(self) -> nn.Module:
        """
        Get the last convolutional layer for Grad-CAM (Stage 13).

        Returns the spatial_projector's Conv2d, which is the last conv
        layer before the global pool. This is the recommended target
        for Grad-CAM attribution.
        """
        return self.spatial_projector[0]  # The Conv2d layer

    @property
    def spatial_output_scale(self) -> int:
        """Spatial downsampling factor from input to spatial features."""
        return self._spatial_scale
