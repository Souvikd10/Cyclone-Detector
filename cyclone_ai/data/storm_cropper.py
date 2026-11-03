"""
Storm-Centered Cropping — Stage 2.

Extracts storm-centered patches from full-disk satellite/environmental data,
producing 512×512 pixel crops (~10° × 10° at 0.05° resolution).
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class StormCropper:
    """
    Crop storm-centered patches from multi-channel gridded data.

    Takes best-track center coordinates and extracts a fixed-size patch
    centered on the storm. Handles edge cases (storm near grid boundary)
    with zero-padding.

    Usage:
        cropper = StormCropper(crop_size=512, field_of_view_deg=10.0)
        patch = cropper.crop(data, center_lat=16.5, center_lon=87.0,
                             lat_range=(0, 30), lon_range=(60, 100))
    """

    def __init__(
        self,
        crop_size: int = 512,
        field_of_view_deg: float = 10.0,
    ):
        """
        Args:
            crop_size: Output patch size in pixels (H = W = crop_size).
            field_of_view_deg: Spatial extent of the crop in degrees (lat and lon).
        """
        self.crop_size = crop_size
        self.fov = field_of_view_deg

    def crop(
        self,
        data: np.ndarray,
        center_lat: float,
        center_lon: float,
        lat_range: tuple[float, float] = (0.0, 30.0),
        lon_range: tuple[float, float] = (60.0, 100.0),
    ) -> np.ndarray:
        """
        Extract a storm-centered patch.

        Args:
            data: Multi-channel gridded data [C, H, W].
            center_lat: Storm center latitude.
            center_lon: Storm center longitude.
            lat_range: (lat_min, lat_max) of the full grid.
            lon_range: (lon_min, lon_max) of the full grid.

        Returns:
            Cropped patch [C, crop_size, crop_size].
        """
        C, H_full, W_full = data.shape

        # Compute pixel coordinates of the storm center
        lat_res = (lat_range[1] - lat_range[0]) / H_full
        lon_res = (lon_range[1] - lon_range[0]) / W_full

        center_y = int((center_lat - lat_range[0]) / lat_res)
        center_x = int((center_lon - lon_range[0]) / lon_res)

        # Compute crop extent in pixels
        half_fov_lat = self.fov / 2.0
        half_fov_lon = self.fov / 2.0
        half_h = int(half_fov_lat / lat_res)
        half_w = int(half_fov_lon / lon_res)

        # Source region bounds (may extend outside grid)
        y_start = center_y - half_h
        y_end = center_y + half_h
        x_start = center_x - half_w
        x_end = center_x + half_w

        # Initialize output with zeros (padding for edges)
        raw_h = y_end - y_start
        raw_w = x_end - x_start
        patch = np.zeros((C, raw_h, raw_w), dtype=data.dtype)

        # Compute valid source and destination ranges
        src_y_start = max(0, y_start)
        src_y_end = min(H_full, y_end)
        src_x_start = max(0, x_start)
        src_x_end = min(W_full, x_end)

        dst_y_start = src_y_start - y_start
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = src_x_start - x_start
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        patch[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = (
            data[:, src_y_start:src_y_end, src_x_start:src_x_end]
        )

        # Resize to target crop_size if needed
        if raw_h != self.crop_size or raw_w != self.crop_size:
            patch = self._resize(patch, self.crop_size)

        return patch

    def crop_with_metadata(
        self,
        data: np.ndarray,
        center_lat: float,
        center_lon: float,
        lat_range: tuple[float, float] = (0.0, 30.0),
        lon_range: tuple[float, float] = (60.0, 100.0),
    ) -> dict:
        """
        Crop with additional metadata about the patch.

        Returns:
            Dict with keys: 'data', 'center_lat', 'center_lon',
                            'patch_lat_range', 'patch_lon_range',
                            'resolution_deg'.
        """
        patch = self.crop(data, center_lat, center_lon, lat_range, lon_range)

        half_fov = self.fov / 2.0
        patch_lat_range = (center_lat - half_fov, center_lat + half_fov)
        patch_lon_range = (center_lon - half_fov, center_lon + half_fov)
        resolution = self.fov / self.crop_size

        return {
            "data": patch,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "patch_lat_range": patch_lat_range,
            "patch_lon_range": patch_lon_range,
            "resolution_deg": resolution,
        }

    @staticmethod
    def _resize(data: np.ndarray, target_size: int) -> np.ndarray:
        """Resize multi-channel data using bilinear interpolation."""
        from scipy.ndimage import zoom

        C, H, W = data.shape
        if H == target_size and W == target_size:
            return data

        zoom_h = target_size / H
        zoom_w = target_size / W
        # Zoom only spatial dimensions, not channel
        resized = zoom(data, (1, zoom_h, zoom_w), order=1)
        return resized.astype(data.dtype)


def create_storm_crops(
    satellite_scenes: list,
    env_snapshots: list,
    track_points: list,
    crop_size: int = 512,
    fov_deg: float = 10.0,
    channel_names: Optional[list[str]] = None,
) -> list[dict]:
    """
    Create storm-centered crops from a sequence of scenes aligned with track data.

    Args:
        satellite_scenes: List of SatelliteScene objects.
        env_snapshots: List of EnvironmentalSnapshot objects.
        track_points: List of TrackPoint objects (for center coordinates).
        crop_size: Output crop size in pixels.
        fov_deg: Field of view in degrees.
        channel_names: Channel names for the combined data.

    Returns:
        List of dicts, each containing 'data' [C, H, W], 'metadata'.
    """
    cropper = StormCropper(crop_size=crop_size, field_of_view_deg=fov_deg)

    if channel_names is None:
        channel_names = ["IR_TIR1", "IR_TIR2", "WV", "VIS", "SST", "WIND_SHEAR", "MOISTURE"]

    crops = []
    for scene, env, point in zip(satellite_scenes, env_snapshots, track_points):
        # Stack satellite and environmental data
        sat_data = scene.to_numpy()  # [C_sat, H, W]
        env_data = env.to_numpy()    # [C_env, H, W]

        # Ensure same spatial dimensions
        if sat_data.shape[1:] != env_data.shape[1:]:
            from cyclone_ai.data.preprocessing import GeoAligner
            aligner = GeoAligner()
            env_data = aligner.regrid(
                env_data,
                source_lat_range=(0, 30), source_lon_range=(60, 100),
                target_lat_range=(0, 30), target_lon_range=(60, 100),
                target_size=sat_data.shape[1:],
            )

        combined = np.concatenate([sat_data, env_data], axis=0)

        # Crop
        crop_result = cropper.crop_with_metadata(
            combined,
            center_lat=point.lat,
            center_lon=point.lon,
        )
        crop_result["timestamp"] = point.timestamp
        crop_result["wind_kt"] = point.wind_kt
        crop_result["pressure_hpa"] = point.pressure_hpa
        crop_result["category"] = point.category
        crop_result["channel_names"] = channel_names

        crops.append(crop_result)

    logger.info(f"Created {len(crops)} storm-centered crops ({crop_size}×{crop_size})")
    return crops
