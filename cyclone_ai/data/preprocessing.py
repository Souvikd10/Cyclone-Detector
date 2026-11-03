"""
Preprocessing & Alignment — Stage 2.

Handles timestamp synchronization, geo-alignment, resampling,
missing-data handling, and normalization across all data sources.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TimeAligner:
    """Snap timestamps from all sources to a common temporal grid."""

    def __init__(self, resolution_minutes: int = 30):
        self.resolution = timedelta(minutes=resolution_minutes)

    def snap_timestamp(self, timestamp: str) -> str:
        """
        Snap an ISO8601 timestamp to the nearest time slot.

        Args:
            timestamp: ISO8601 timestamp string.

        Returns:
            Snapped ISO8601 timestamp.
        """
        dt = self._parse_timestamp(timestamp)
        # Round to nearest resolution
        seconds = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        slot_seconds = self.resolution.total_seconds()
        snapped_seconds = round(seconds / slot_seconds) * slot_seconds
        snapped = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            seconds=snapped_seconds
        )
        return snapped.isoformat()

    def generate_time_grid(self, start: str, end: str) -> list[str]:
        """Generate a regular time grid between start and end."""
        start_dt = self._parse_timestamp(start)
        end_dt = self._parse_timestamp(end)

        timestamps = []
        current = start_dt
        while current <= end_dt:
            timestamps.append(current.isoformat())
            current += self.resolution

        return timestamps

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse ISO8601 timestamp string."""
        ts = ts.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            # Fallback for older Python
            return datetime.strptime(ts.split("+")[0], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc
            )


class GeoAligner:
    """Regrid all data sources to a common spatial grid."""

    def __init__(self, target_resolution_deg: float = 0.05):
        self.resolution = target_resolution_deg

    def regrid(
        self,
        data: np.ndarray,
        source_lat_range: tuple[float, float],
        source_lon_range: tuple[float, float],
        target_lat_range: tuple[float, float],
        target_lon_range: tuple[float, float],
        target_size: tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Regrid data from source grid to target grid using bilinear interpolation.

        Args:
            data: Input array [H_src, W_src] or [C, H_src, W_src].
            source_lat_range: (lat_min, lat_max) of source grid.
            source_lon_range: (lon_min, lon_max) of source grid.
            target_lat_range: (lat_min, lat_max) of target grid.
            target_lon_range: (lon_min, lon_max) of target grid.
            target_size: (H_target, W_target) output dimensions.

        Returns:
            Regridded array of shape [H_target, W_target] or [C, H_target, W_target].
        """
        from scipy.interpolate import RegularGridInterpolator

        has_channels = data.ndim == 3

        if has_channels:
            C = data.shape[0]
            result = np.zeros((C, *target_size), dtype=np.float32)
            for c in range(C):
                result[c] = self._regrid_2d(
                    data[c], source_lat_range, source_lon_range,
                    target_lat_range, target_lon_range, target_size,
                )
            return result
        else:
            return self._regrid_2d(
                data, source_lat_range, source_lon_range,
                target_lat_range, target_lon_range, target_size,
            )

    def _regrid_2d(
        self,
        data: np.ndarray,
        src_lat: tuple, src_lon: tuple,
        tgt_lat: tuple, tgt_lon: tuple,
        tgt_size: tuple,
    ) -> np.ndarray:
        """Regrid a single 2D field."""
        from scipy.interpolate import RegularGridInterpolator

        H_src, W_src = data.shape
        src_lat_grid = np.linspace(src_lat[0], src_lat[1], H_src)
        src_lon_grid = np.linspace(src_lon[0], src_lon[1], W_src)

        interp = RegularGridInterpolator(
            (src_lat_grid, src_lon_grid), data,
            method="linear", bounds_error=False, fill_value=None,
        )

        H_tgt, W_tgt = tgt_size
        tgt_lat_grid = np.linspace(tgt_lat[0], tgt_lat[1], H_tgt)
        tgt_lon_grid = np.linspace(tgt_lon[0], tgt_lon[1], W_tgt)
        tgt_lon_mesh, tgt_lat_mesh = np.meshgrid(tgt_lon_grid, tgt_lat_grid)
        points = np.stack([tgt_lat_mesh.ravel(), tgt_lon_mesh.ravel()], axis=-1)

        result = interp(points).reshape(H_tgt, W_tgt)
        return result.astype(np.float32)


class Normalizer:
    """Per-channel z-score normalization with running statistics."""

    def __init__(self):
        self.channel_stats: dict[str, dict[str, float]] = {}

    def fit(self, data: np.ndarray, channel_names: list[str]):
        """
        Compute per-channel mean and std from data.

        Args:
            data: [C, H, W] or [N, C, H, W] array.
            channel_names: Names for each channel.
        """
        if data.ndim == 3:
            data = data[np.newaxis]  # [1, C, H, W]

        for i, name in enumerate(channel_names):
            channel_data = data[:, i]
            self.channel_stats[name] = {
                "mean": float(np.nanmean(channel_data)),
                "std": float(np.nanstd(channel_data)),
            }

        logger.info(f"Normalizer fitted on {len(channel_names)} channels")

    def transform(self, data: np.ndarray, channel_names: list[str]) -> np.ndarray:
        """
        Apply z-score normalization.

        Args:
            data: [C, H, W] array.
            channel_names: Names for each channel (must match fit).

        Returns:
            Normalized array of same shape.
        """
        result = np.copy(data).astype(np.float32)
        for i, name in enumerate(channel_names):
            if name in self.channel_stats:
                stats = self.channel_stats[name]
                std = stats["std"] if stats["std"] > 1e-8 else 1.0
                result[i] = (result[i] - stats["mean"]) / std
            else:
                logger.warning(f"No stats for channel '{name}', skipping normalization")
        return result

    def fit_transform(self, data: np.ndarray, channel_names: list[str]) -> np.ndarray:
        """Fit statistics and normalize in one step."""
        self.fit(data, channel_names)
        return self.transform(data, channel_names)

    def get_stats(self) -> dict:
        """Return channel statistics."""
        return dict(self.channel_stats)


class MissingDataHandler:
    """Handle missing or corrupted data in satellite/environmental fields."""

    def __init__(self, max_missing_frac: float = 0.3):
        self.max_missing_frac = max_missing_frac

    def handle(self, data: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, dict]:
        """
        Handle missing data in a multi-channel array.

        Strategy:
        - Small gaps (< max_missing_frac): spatial interpolation
        - Large gaps (>= max_missing_frac): flag channel as degraded

        Args:
            data: [C, H, W] array (NaN where data is missing).
            channel_names: Names for each channel.

        Returns:
            Tuple of (cleaned data, quality_info dict).
        """
        result = np.copy(data)
        quality_info = {}

        for i, name in enumerate(channel_names):
            channel = result[i]
            nan_mask = np.isnan(channel)
            missing_frac = float(nan_mask.sum()) / channel.size

            quality_info[name] = {
                "missing_fraction": missing_frac,
                "usable": missing_frac < self.max_missing_frac,
                "interpolated": False,
            }

            if missing_frac == 0:
                continue

            if missing_frac < self.max_missing_frac:
                # Spatial interpolation for small gaps
                result[i] = self._interpolate_gaps(channel)
                quality_info[name]["interpolated"] = True
                logger.debug(f"Channel '{name}': interpolated {missing_frac:.1%} missing data")
            else:
                # Fill with channel mean for large gaps (will be downweighted by reliability)
                channel_mean = np.nanmean(channel) if np.any(~nan_mask) else 0.0
                result[i] = np.where(nan_mask, channel_mean, channel)
                logger.warning(
                    f"Channel '{name}': {missing_frac:.1%} missing — flagged as degraded"
                )

        return result, quality_info

    @staticmethod
    def _interpolate_gaps(data: np.ndarray) -> np.ndarray:
        """Interpolate NaN gaps using nearest-neighbor then Gaussian smoothing."""
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        nan_mask = np.isnan(data)
        if not nan_mask.any():
            return data

        # Nearest-neighbor fill
        filled = np.copy(data)
        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        filled = data[tuple(indices)]

        # Light Gaussian smoothing only at filled locations
        smoothed = gaussian_filter(filled, sigma=2.0)
        result = np.where(nan_mask, smoothed, data)

        return result.astype(np.float32)


def preprocess_scene(
    satellite_data: np.ndarray,
    env_data: np.ndarray,
    channel_names: list[str],
    normalizer: Optional[Normalizer] = None,
    fit_normalizer: bool = True,
) -> tuple[np.ndarray, Normalizer, dict]:
    """
    Full preprocessing pipeline for a single scene.

    1. Handle missing data
    2. Concatenate satellite + environmental channels
    3. Normalize

    Args:
        satellite_data: [C_sat, H, W] satellite imagery.
        env_data: [C_env, H, W] environmental fields.
        channel_names: Names for all channels (satellite + env).
        normalizer: Pre-fitted normalizer (None to create new).
        fit_normalizer: Whether to fit the normalizer on this data.

    Returns:
        Tuple of (normalized data [C, H, W], normalizer, quality info).
    """
    # Concatenate all channels
    combined = np.concatenate([satellite_data, env_data], axis=0)

    # Handle missing data
    handler = MissingDataHandler()
    cleaned, quality = handler.handle(combined, channel_names)

    # Normalize
    if normalizer is None:
        normalizer = Normalizer()

    if fit_normalizer:
        normalized = normalizer.fit_transform(cleaned, channel_names)
    else:
        normalized = normalizer.transform(cleaned, channel_names)

    return normalized, normalizer, quality
