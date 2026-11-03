"""
Detection Head — Stage 5.

Cyclone detection: presence classification, bounding box regression,
center localization, and segmentation mask generation.

Takes spatial feature maps from the shared encoder (Stage 4).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FPNDecoder(nn.Module):
    """
    Lightweight Feature Pyramid Network decoder for segmentation mask.

    Upsamples encoder spatial features back to input resolution and
    produces a binary segmentation mask (cyclone cloud vs. background).
    """

    def __init__(self, in_channels: int = 768, mid_channels: int = 256, out_channels: int = 1):
        super().__init__()

        # Progressive upsampling with skip refinement
        self.reduce = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(mid_channels)

        self.up1 = self._upsample_block(mid_channels, mid_channels // 2)   # 2×
        self.up2 = self._upsample_block(mid_channels // 2, mid_channels // 4)  # 4×
        self.up3 = self._upsample_block(mid_channels // 4, mid_channels // 8)  # 8×
        self.up4 = self._upsample_block(mid_channels // 8, mid_channels // 16) # 16×

        self.final = nn.Sequential(
            nn.Conv2d(mid_channels // 16, mid_channels // 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 16),
            nn.GELU(),
            nn.Conv2d(mid_channels // 16, out_channels, 1),
        )

    def _upsample_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, H', W'] spatial features from encoder.
            target_size: Optional (H, W) target output resolution.

        Returns:
            [B, 1, H, W] binary mask logits.
        """
        x = F.gelu(self.bn0(self.reduce(x)))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)

        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        return x


class DetectionHead(nn.Module):
    """
    Cyclone Detection Head (Stage 5).

    Produces:
    - present: boolean cyclone presence
    - confidence: detection confidence score [0, 1]
    - center: (lat, lon) storm center coordinates
    - bbox: [lat_min, lon_min, lat_max, lon_max] bounding box
    - mask: binary segmentation mask

    Input: spatial feature map [B, 768, H', W'] from SharedEncoder.

    Args:
        feature_dim: Spatial feature channels from encoder.
        mask_threshold: Threshold for binary mask (at inference).
        min_confidence: Minimum confidence to report detection.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        mask_threshold: float = 0.5,
        min_confidence: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.mask_threshold = mask_threshold
        self.min_confidence = min_confidence

        # Global average pool for scalar predictions
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Presence classifier
        self.presence_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        # Center regression (lat, lon)
        self.center_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # (lat, lon) normalized
        )

        # BBox regression (4 coords)
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),  # (lat_min, lon_min, lat_max, lon_max) normalized
        )

        # Segmentation mask decoder
        self.mask_decoder = FPNDecoder(
            in_channels=feature_dim,
            mid_channels=256,
            out_channels=1,
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        target_mask_size: Optional[tuple] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            spatial_features: [B, C, H', W'] from SharedEncoder.
            target_mask_size: (H, W) for mask output. Default: 512×512.

        Returns:
            Dict with keys:
                'presence_logit': [B, 1] raw logit
                'confidence': [B, 1] sigmoid confidence
                'center': [B, 2] normalized (lat, lon)
                'bbox': [B, 4] normalized bbox coords
                'mask_logit': [B, 1, H, W] raw mask logits
                'mask': [B, 1, H, W] binary mask (thresholded, detached)
        """
        if target_mask_size is None:
            target_mask_size = (512, 512)

        # Global features for scalar predictions
        pooled = self.gap(spatial_features).squeeze(-1).squeeze(-1)  # [B, C]

        # Presence
        presence_logit = self.presence_head(pooled)  # [B, 1]
        confidence = torch.sigmoid(presence_logit)    # [B, 1]

        # Center
        center = self.center_head(pooled)  # [B, 2]

        # BBox
        bbox = self.bbox_head(pooled)  # [B, 4]

        # Mask
        mask_logit = self.mask_decoder(spatial_features, target_size=target_mask_size)
        mask = (torch.sigmoid(mask_logit) > self.mask_threshold).float().detach()

        return {
            "presence_logit": presence_logit,
            "confidence": confidence,
            "center": center,
            "bbox": bbox,
            "mask_logit": mask_logit,
            "mask": mask,
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute detection losses.

        Args:
            predictions: Output from forward().
            targets: Dict with 'present' [B,1], 'center' [B,2],
                     'bbox' [B,4], 'mask' [B,1,H,W].

        Returns:
            Dict of individual losses + 'total'.
        """
        losses = {}

        # Presence loss (BCE)
        losses["presence"] = F.binary_cross_entropy_with_logits(
            predictions["presence_logit"],
            targets["present"].float(),
        )

        # Center loss (Smooth L1) — only for present cyclones
        present_mask = targets["present"].squeeze(-1).bool()
        if present_mask.any():
            losses["center"] = F.smooth_l1_loss(
                predictions["center"][present_mask],
                targets["center"][present_mask],
            )

            # BBox loss (Smooth L1)
            losses["bbox"] = F.smooth_l1_loss(
                predictions["bbox"][present_mask],
                targets["bbox"][present_mask],
            )

            # Mask loss (Dice + BCE)
            pred_mask = predictions["mask_logit"][present_mask]
            tgt_mask = targets["mask"][present_mask]
            losses["mask_bce"] = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask)
            losses["mask_dice"] = self._dice_loss(pred_mask, tgt_mask)
            losses["mask"] = losses["mask_bce"] + losses["mask_dice"]
        else:
            losses["center"] = torch.tensor(0.0, device=predictions["center"].device)
            losses["bbox"] = torch.tensor(0.0, device=predictions["center"].device)
            losses["mask"] = torch.tensor(0.0, device=predictions["center"].device)

        losses["total"] = (
            losses["presence"]
            + losses["center"]
            + losses["bbox"]
            + losses["mask"]
        )

        return losses

    @staticmethod
    def _dice_loss(pred_logit: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Dice loss for segmentation."""
        pred = torch.sigmoid(pred_logit)
        intersection = (pred * target).sum(dim=(-2, -1))
        union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return (1.0 - dice).mean()

    def to_contract(
        self,
        predictions: dict[str, torch.Tensor],
        crop_metadata: dict,
        mask_url: str,
    ) -> dict:
        """
        Convert raw predictions to contract-compliant detection dict.

        Args:
            predictions: Output from forward().
            crop_metadata: Contains 'center_lat', 'center_lon', 'patch_lat_range', etc.
            mask_url: MinIO URL of uploaded mask image.

        Returns:
            Contract detection dict.
        """
        conf = float(predictions["confidence"][0, 0].cpu())
        present = conf >= self.min_confidence

        # Convert normalized center to absolute coords
        center_norm = predictions["center"][0].cpu().numpy()
        lat_range = crop_metadata.get("patch_lat_range", (0, 30))
        lon_range = crop_metadata.get("patch_lon_range", (60, 100))
        center_lat = lat_range[0] + center_norm[0] * (lat_range[1] - lat_range[0])
        center_lon = lon_range[0] + center_norm[1] * (lon_range[1] - lon_range[0])

        # Convert normalized bbox
        bbox_norm = predictions["bbox"][0].cpu().numpy()
        bbox = [
            float(lat_range[0] + bbox_norm[0] * (lat_range[1] - lat_range[0])),
            float(lon_range[0] + bbox_norm[1] * (lon_range[1] - lon_range[0])),
            float(lat_range[0] + bbox_norm[2] * (lat_range[1] - lat_range[0])),
            float(lon_range[0] + bbox_norm[3] * (lon_range[1] - lon_range[0])),
        ]

        return {
            "present": present,
            "confidence": round(conf, 4),
            "center": {"lat": round(float(center_lat), 4), "lon": round(float(center_lon), 4)},
            "bbox": [round(b, 4) for b in bbox],
            "mask_url": mask_url,
        }
