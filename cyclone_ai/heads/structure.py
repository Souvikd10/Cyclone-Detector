"""
Structural Characterization Head — Stage 6.

Eye detection, center localization, diameter estimation,
symmetry scoring, and circularity analysis.

Uses spatial attention pooling to focus on the eye region rather than
global average pooling — critical because the eye is typically a small
fraction of the total cyclone cloud system.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpatialAttentionPool(nn.Module):
    """
    Learnable spatial attention pooling.

    Instead of global average pooling (which treats all spatial locations
    equally), this module learns to attend to specific regions — allowing
    the model to focus on the eye area for structural analysis.

    Args:
        feature_dim: Input feature channels.
        num_heads: Number of attention heads.
    """

    def __init__(self, feature_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.query = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.key_proj = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] spatial features.

        Returns:
            Tuple of:
                pooled: [B, C] attention-pooled features
                attn_map: [B, num_heads, H*W] attention weights (for visualization)
        """
        B, C, H, W = x.shape

        # Project keys and values
        K = self.key_proj(x).reshape(B, self.num_heads, self.head_dim, H * W)    # [B, heads, d, HW]
        V = self.value_proj(x).reshape(B, self.num_heads, self.head_dim, H * W)  # [B, heads, d, HW]

        # Attention: query attends to all spatial locations
        Q = self.query.expand(B, -1, -1, -1)  # [B, heads, 1, d]
        attn = torch.matmul(Q, K) / math.sqrt(self.head_dim)  # [B, heads, 1, HW]
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.matmul(attn, V.transpose(-2, -1))  # [B, heads, 1, d]
        out = out.reshape(B, C)  # [B, C]
        out = self.out_proj(out)
        out = self.norm(out)

        attn_map = attn.squeeze(2)  # [B, heads, HW]
        return out, attn_map


class StructureHead(nn.Module):
    """
    Structural Characterization Head (Stage 6).

    Analyzes the cyclone's structural properties:
    - Eye presence and confidence
    - Eye center (lat, lon)
    - Eye diameter (km)
    - Symmetry score (0–1): azimuthal variance of cloud structure
    - Circularity score (0–1): how circular the cyclone is

    Input: spatial feature map [B, 768, H', W'] from SharedEncoder.

    Args:
        feature_dim: Spatial feature channels.
        hidden_dim: Hidden layer size for MLPs.
        num_attention_heads: Heads for spatial attention pooling.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_attention_heads: int = 4,
        min_eye_confidence: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.min_eye_confidence = min_eye_confidence

        # Spatial attention pooling (focuses on eye region)
        self.attention_pool = SpatialAttentionPool(
            feature_dim=feature_dim,
            num_heads=num_attention_heads,
        )

        # Eye presence classifier (focal loss for class imbalance)
        self.eye_presence = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        # Eye center regression
        self.eye_center = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # (lat, lon) normalized
            nn.Sigmoid(),             # constrain to [0, 1]
        )

        # Eye diameter regression (km)
        self.eye_diameter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),  # diameter must be positive
        )

        # Symmetry and circularity from spatial features
        # Use both attention-pooled and global features
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.symmetry_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # constrain to [0, 1]
        )

        self.circularity_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # constrain to [0, 1]
        )

    def forward(self, spatial_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            spatial_features: [B, C, H', W'] from SharedEncoder.

        Returns:
            Dict with keys:
                'eye_logit': [B, 1] raw logit for eye presence
                'eye_confidence': [B, 1] sigmoid confidence
                'eye_center': [B, 2] normalized (lat, lon)
                'eye_diameter_km': [B, 1] diameter in km
                'symmetry': [B, 1] symmetry score [0, 1]
                'circularity': [B, 1] circularity score [0, 1]
                'attention_map': [B, heads, H'*W'] spatial attention weights
        """
        # Attention-pooled features (eye-focused)
        attn_pooled, attn_map = self.attention_pool(spatial_features)  # [B, C], [B, heads, HW]

        # Global features
        global_pooled = self.gap(spatial_features).squeeze(-1).squeeze(-1)  # [B, C]

        # Eye presence
        eye_logit = self.eye_presence(attn_pooled)  # [B, 1]
        eye_confidence = torch.sigmoid(eye_logit)    # [B, 1]

        # Eye center
        eye_center = self.eye_center(attn_pooled)  # [B, 2]

        # Eye diameter (scale to realistic range: 0–100 km)
        eye_diameter_raw = self.eye_diameter(attn_pooled)  # [B, 1]
        eye_diameter_km = eye_diameter_raw * 100.0  # scale factor

        # Symmetry and circularity use both attention and global features
        combined = torch.cat([attn_pooled, global_pooled], dim=-1)  # [B, 2*C]
        symmetry = self.symmetry_head(combined)    # [B, 1]
        circularity = self.circularity_head(combined)  # [B, 1]

        return {
            "eye_logit": eye_logit,
            "eye_confidence": eye_confidence,
            "eye_center": eye_center,
            "eye_diameter_km": eye_diameter_km,
            "symmetry": symmetry,
            "circularity": circularity,
            "attention_map": attn_map,
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute structure losses.

        Uses focal loss for eye presence (handles class imbalance).

        Args:
            predictions: Output from forward().
            targets: Dict with 'eye_present' [B,1], 'eye_center' [B,2],
                     'eye_diameter_km' [B,1], 'symmetry' [B,1], 'circularity' [B,1].

        Returns:
            Dict of individual losses + 'total'.
        """
        losses = {}

        # Eye presence — focal loss
        losses["eye_presence"] = self._focal_loss(
            predictions["eye_logit"],
            targets["eye_present"].float(),
            alpha=0.75,  # weight for positive class (eye present)
            gamma=2.0,
        )

        # Only compute eye regression losses for storms with eyes
        eye_mask = targets["eye_present"].squeeze(-1).bool()

        if eye_mask.any():
            losses["eye_center"] = F.smooth_l1_loss(
                predictions["eye_center"][eye_mask],
                targets["eye_center"][eye_mask],
            )
            losses["eye_diameter"] = F.mse_loss(
                predictions["eye_diameter_km"][eye_mask],
                targets["eye_diameter_km"][eye_mask],
            )
        else:
            dev = predictions["eye_center"].device
            losses["eye_center"] = torch.tensor(0.0, device=dev)
            losses["eye_diameter"] = torch.tensor(0.0, device=dev)

        # Symmetry and circularity (always computed)
        losses["symmetry"] = F.mse_loss(predictions["symmetry"], targets["symmetry"])
        losses["circularity"] = F.mse_loss(predictions["circularity"], targets["circularity"])

        losses["total"] = (
            losses["eye_presence"]
            + losses["eye_center"]
            + 0.1 * losses["eye_diameter"]  # smaller weight (km scale)
            + losses["symmetry"]
            + losses["circularity"]
        )

        return losses

    @staticmethod
    def _focal_loss(
        pred_logit: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance in eye detection."""
        bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none")
        prob = torch.sigmoid(pred_logit)
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** gamma
        return (focal_weight * bce).mean()

    def to_contract(
        self,
        predictions: dict[str, torch.Tensor],
        crop_metadata: dict,
    ) -> dict:
        """Convert raw predictions to contract-compliant structure dict."""
        eye_conf = float(predictions["eye_confidence"][0, 0].cpu())
        eye_present = eye_conf >= self.min_eye_confidence

        # Convert normalized eye center to absolute coords
        eye_center_norm = predictions["eye_center"][0].cpu().numpy()
        lat_range = crop_metadata.get("patch_lat_range", (0, 30))
        lon_range = crop_metadata.get("patch_lon_range", (60, 100))

        if eye_present:
            eye_lat = lat_range[0] + eye_center_norm[0] * (lat_range[1] - lat_range[0])
            eye_lon = lon_range[0] + eye_center_norm[1] * (lon_range[1] - lon_range[0])
        else:
            eye_lat, eye_lon = 0.0, 0.0

        return {
            "eye_present": eye_present,
            "eye_confidence": round(eye_conf, 4),
            "eye_center": {"lat": round(float(eye_lat), 4), "lon": round(float(eye_lon), 4)},
            "eye_diameter_km": round(float(predictions["eye_diameter_km"][0, 0].cpu()), 2),
            "symmetry": round(float(predictions["symmetry"][0, 0].cpu()), 4),
            "circularity": round(float(predictions["circularity"][0, 0].cpu()), 4),
        }
