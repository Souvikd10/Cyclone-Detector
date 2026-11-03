"""
Track + Intensity Forecast Model — Stage 9.

Transformer decoder that predicts future cyclone track positions and
intensity at multiple forecast horizons (+6h, +12h, +24h, +48h, +72h).

Uses learned horizon queries so each forecast time specializes independently.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HaversineLoss(nn.Module):
    """
    Great-circle distance loss for track forecast evaluation.

    Computes the haversine distance (in km) between predicted and
    actual lat/lon positions, used as a differentiable loss.
    """

    EARTH_RADIUS_KM = 6371.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, N, 2] predicted (lat, lon) in degrees.
            target: [B, N, 2] target (lat, lon) in degrees.

        Returns:
            Scalar mean haversine distance loss (km).
        """
        lat1 = torch.deg2rad(pred[..., 0])
        lon1 = torch.deg2rad(pred[..., 1])
        lat2 = torch.deg2rad(target[..., 0])
        lon2 = torch.deg2rad(target[..., 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        a = torch.clamp(a, 0, 1)  # numerical stability
        c = 2 * torch.arcsin(torch.sqrt(a))

        distance_km = self.EARTH_RADIUS_KM * c
        return distance_km.mean()


class PinballLoss(nn.Module):
    """
    Pinball (quantile) loss for probabilistic intensity forecasting.

    Penalizes over/under-prediction asymmetrically based on the quantile.
    """

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - pred
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error,
        )
        return loss.mean()


class ForecastModel(nn.Module):
    """
    Track + Intensity Forecast Model (Stage 9).

    Uses a Transformer decoder with learned horizon queries to predict
    future cyclone positions and intensity at multiple lead times.

    Architecture:
        Context = [temporal_features || env_scalars] → Linear → 512-d
        Queries = 5 learned embeddings (one per forecast horizon)
        Decoder = TransformerDecoder(d_model=512, heads=8, layers=4)
        Track head: Linear(512, 3) → (Δlat, Δlon, cone_radius_km)
        Intensity head: Linear(512, 3) → (wind_kt, interval_low, interval_high)

    Args:
        temporal_dim: Temporal feature dimension.
        env_dim: Environmental scalar feature dimension.
        model_dim: Transformer model dimension.
        num_heads: Attention heads.
        num_layers: Decoder layers.
        horizons: List of forecast horizons in hours.
    """

    def __init__(
        self,
        temporal_dim: int = 512,
        env_dim: int = 8,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        horizons: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizons = horizons or [6, 12, 24, 48, 72]
        self.num_horizons = len(self.horizons)
        self.model_dim = model_dim

        # Context projection
        self.context_proj = nn.Sequential(
            nn.Linear(temporal_dim + env_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned horizon queries (one per forecast time)
        self.horizon_queries = nn.Parameter(
            torch.randn(self.num_horizons, model_dim) * 0.02
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers,
        )

        # Track prediction head: (Δlat, Δlon, cone_radius_km)
        self.track_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 3),
        )

        # Intensity prediction head: (wind_kt, interval_low, interval_high)
        self.intensity_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 3),
        )

        # Losses
        self.haversine_loss = HaversineLoss()
        self.pinball_low = PinballLoss(quantile=0.1)
        self.pinball_high = PinballLoss(quantile=0.9)

    def forward(
        self,
        temporal_features: torch.Tensor,
        env_scalars: torch.Tensor,
        current_position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            temporal_features: [B, temporal_dim] from TemporalBlock.
            env_scalars: [B, env_dim] environmental scalars.
            current_position: [B, 2] current (lat, lon) for absolute positioning.

        Returns:
            Dict with:
                'track_delta': [B, N_horizons, 3] (Δlat, Δlon, cone_radius_km)
                'track_absolute': [B, N_horizons, 2] absolute (lat, lon) if current_position given
                'intensity': [B, N_horizons, 3] (wind_kt, interval_low, interval_high)
                'cone_radius': [B, N_horizons] cone radius in km
        """
        B = temporal_features.shape[0]

        # Build context
        context = torch.cat([temporal_features, env_scalars], dim=-1)  # [B, temporal+env]
        context = self.context_proj(context)  # [B, model_dim]
        context = context.unsqueeze(1)  # [B, 1, model_dim] — memory for decoder

        # Expand queries for batch
        queries = self.horizon_queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, model_dim]

        # Decode
        decoded = self.decoder(queries, context)  # [B, N, model_dim]

        # Predict track and intensity
        track_raw = self.track_head(decoded)       # [B, N, 3]
        intensity_raw = self.intensity_head(decoded)  # [B, N, 3]

        # Split track outputs
        track_delta = track_raw[..., :2]           # [B, N, 2] Δlat, Δlon
        cone_radius = F.relu(track_raw[..., 2])    # [B, N] cone radius (must be positive)

        # Compute absolute positions
        track_absolute = None
        if current_position is not None:
            track_absolute = current_position.unsqueeze(1) + track_delta  # [B, N, 2]

        # Split intensity outputs
        wind_kt = F.relu(intensity_raw[..., 0])            # [B, N] non-negative wind
        interval_low = F.relu(intensity_raw[..., 1])       # [B, N]
        interval_high = F.relu(intensity_raw[..., 2])      # [B, N]

        # Ensure interval_low <= wind_kt <= interval_high
        interval_low = torch.min(interval_low, wind_kt)
        interval_high = torch.max(interval_high, wind_kt)

        return {
            "track_delta": track_delta,
            "track_absolute": track_absolute,
            "cone_radius": cone_radius,
            "wind_kt": wind_kt,
            "interval_low": interval_low,
            "interval_high": interval_high,
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute forecast losses.

        Args:
            targets: Dict with 'track' [B, N, 2] (lat, lon),
                     'wind_kt' [B, N], 'cone_radius' [B, N].
        """
        losses = {}

        # Track loss (Haversine)
        if predictions["track_absolute"] is not None:
            losses["track"] = self.haversine_loss(
                predictions["track_absolute"], targets["track"]
            )
        else:
            losses["track"] = F.mse_loss(
                predictions["track_delta"], targets.get("track_delta", targets["track"])
            )

        # Wind speed MSE
        losses["wind"] = F.mse_loss(predictions["wind_kt"], targets["wind_kt"])

        # Cone radius MSE
        if "cone_radius" in targets:
            losses["cone"] = F.mse_loss(predictions["cone_radius"], targets["cone_radius"])
        else:
            losses["cone"] = torch.tensor(0.0, device=predictions["wind_kt"].device)

        # Quantile losses for interval
        losses["quantile_low"] = self.pinball_low(
            predictions["interval_low"], targets["wind_kt"]
        )
        losses["quantile_high"] = self.pinball_high(
            predictions["interval_high"], targets["wind_kt"]
        )

        losses["total"] = (
            2.0 * losses["track"]
            + losses["wind"]
            + 0.5 * losses["cone"]
            + 0.5 * (losses["quantile_low"] + losses["quantile_high"])
        )

        return losses

    def to_contract(
        self,
        predictions: dict[str, torch.Tensor],
    ) -> dict:
        """Convert to contract-compliant forecast dict."""
        track_list = []
        intensity_list = []

        for i, h in enumerate(self.horizons):
            # Track
            track_entry = {"t": f"+{h}h"}
            if predictions["track_absolute"] is not None:
                track_entry["lat"] = round(float(predictions["track_absolute"][0, i, 0].detach().cpu()), 2)
                track_entry["lon"] = round(float(predictions["track_absolute"][0, i, 1].detach().cpu()), 2)
            else:
                track_entry["lat"] = round(float(predictions["track_delta"][0, i, 0].detach().cpu()), 2)
                track_entry["lon"] = round(float(predictions["track_delta"][0, i, 1].detach().cpu()), 2)
            track_entry["cone_radius_km"] = round(float(predictions["cone_radius"][0, i].detach().cpu()), 1)
            track_list.append(track_entry)

            # Intensity
            intensity_list.append({
                "t": f"+{h}h",
                "wind_kt": round(float(predictions["wind_kt"][0, i].detach().cpu()), 1),
                "interval": [
                    round(float(predictions["interval_low"][0, i].detach().cpu()), 1),
                    round(float(predictions["interval_high"][0, i].detach().cpu()), 1),
                ],
            })

        return {"track": track_list, "intensity": intensity_list}
