"""
ERA5 Client — Environmental reanalysis data acquisition.

Stage 1: Data Acquisition (environmental fields component).

Provides sea surface temperature (SST), vertical wind shear,
relative humidity, and ocean heat content fields. Currently generates
synthetic data with realistic value ranges; swap-ready for CDS API.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentalField:
    """A single environmental variable on a lat/lon grid."""

    variable: str
    data: np.ndarray           # [H, W] grid
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    resolution_deg: float
    timestamp: str
    units: str


@dataclass
class EnvironmentalSnapshot:
    """Collection of environmental fields at a single timestamp."""

    timestamp: str
    storm_id: str
    fields: dict[str, EnvironmentalField]  # variable name → field

    def to_numpy(self) -> np.ndarray:
        """Stack all fields into [C, H, W] array."""
        arrays = []
        for name in sorted(self.fields.keys()):
            arrays.append(self.fields[name].data)
        return np.stack(arrays, axis=0).astype(np.float32)

    def get_scalar_features(self, center_lat: float, center_lon: float,
                            radius_deg: float = 2.0) -> dict[str, float]:
        """
        Extract scalar environmental features around storm center.

        Computes mean values within a radius around the center.
        These are used as inputs to the RI model (Stage 10).
        """
        features = {}
        for name, field in self.fields.items():
            H, W = field.data.shape
            lat_grid = np.linspace(field.lat_range[0], field.lat_range[1], H)
            lon_grid = np.linspace(field.lon_range[0], field.lon_range[1], W)
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

            dist = np.sqrt((lat_mesh - center_lat) ** 2 + (lon_mesh - center_lon) ** 2)
            mask = dist <= radius_deg

            if mask.any():
                features[f"{name}_mean"] = float(np.mean(field.data[mask]))
                features[f"{name}_std"] = float(np.std(field.data[mask]))
            else:
                features[f"{name}_mean"] = float(np.mean(field.data))
                features[f"{name}_std"] = float(np.std(field.data))

        return features


class SyntheticEnvironmentalGenerator:
    """
    Generates synthetic environmental fields with realistic spatial patterns.

    Value ranges are based on typical North Indian Ocean conditions during
    tropical cyclone season (April–December).
    """

    VARIABLE_PARAMS = {
        "SST": {
            "units": "°C",
            "base_value": 29.0,       # warm tropical ocean
            "lat_gradient": -0.15,     # cooler at higher latitudes
            "cyclone_effect": -0.5,    # slight cooling from upwelling
            "noise_std": 0.3,
        },
        "WIND_SHEAR": {
            "units": "m/s",
            "base_value": 8.0,         # moderate shear
            "lat_gradient": 0.3,       # more shear at higher latitudes
            "cyclone_effect": -3.0,    # low shear near cyclone (favorable)
            "noise_std": 2.0,
        },
        "MOISTURE": {
            "units": "%",
            "base_value": 65.0,        # mid-level RH
            "lat_gradient": -0.5,
            "cyclone_effect": 15.0,    # high moisture near cyclone
            "noise_std": 5.0,
        },
    }

    def __init__(self, grid_size: int = 512, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

    def generate_snapshot(
        self,
        storm_id: str,
        timestamp: str,
        center_lat: float = 15.0,
        center_lon: float = 85.0,
        lat_range: tuple[float, float] = (0.0, 30.0),
        lon_range: tuple[float, float] = (60.0, 100.0),
    ) -> EnvironmentalSnapshot:
        """Generate a complete environmental snapshot."""
        snapshot = EnvironmentalSnapshot(
            timestamp=timestamp,
            storm_id=storm_id,
            fields={},
        )

        resolution = (lat_range[1] - lat_range[0]) / self.grid_size

        for var_name, params in self.VARIABLE_PARAMS.items():
            field_data = self._generate_field(
                params, center_lat, center_lon, lat_range, lon_range,
            )
            snapshot.fields[var_name] = EnvironmentalField(
                variable=var_name,
                data=field_data,
                lat_range=lat_range,
                lon_range=lon_range,
                resolution_deg=resolution,
                timestamp=timestamp,
                units=params["units"],
            )

        return snapshot

    def _generate_field(
        self,
        params: dict,
        center_lat: float,
        center_lon: float,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
    ) -> np.ndarray:
        """Generate a single environmental field."""
        H = W = self.grid_size

        lat_grid = np.linspace(lat_range[0], lat_range[1], H)
        lon_grid = np.linspace(lon_range[0], lon_range[1], W)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Base field with latitude gradient
        base = params["base_value"] + params["lat_gradient"] * (lat_mesh - np.mean(lat_range))

        # Cyclone effect — radial pattern centered on storm
        r = np.sqrt((lat_mesh - center_lat) ** 2 + (lon_mesh - center_lon) ** 2)
        cyclone_influence = params["cyclone_effect"] * np.exp(-r ** 2 / (2 * 3.0 ** 2))

        # Add noise
        noise = self.rng.normal(0, params["noise_std"], (H, W))

        field = base + cyclone_influence + noise
        return field.astype(np.float32)


class ERA5Client:
    """
    Client for ERA5/CDS environmental reanalysis data.

    Currently uses synthetic data generation. Designed for drop-in replacement
    with real CDS API access when credentials are available.

    Usage:
        client = ERA5Client(config_path="configs/data_sources.yaml")
        snapshot = client.get_snapshot("2020121N10087", "2020-05-17T06:00:00Z",
                                        center_lat=12.5, center_lon=85.8)
        sst_field = snapshot.fields["SST"]
        scalars = snapshot.get_scalar_features(12.5, 85.8)
    """

    def __init__(self, config_path: str = "configs/data_sources.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.era5_config = config.get("era5", {})
        self.synthetic_config = config.get("synthetic", {})
        self.data_mode = config.get("data_mode", "synthetic")

        grid_size = self.synthetic_config.get("image_size", 512)
        self._generator = SyntheticEnvironmentalGenerator(grid_size=grid_size)

        logger.info(f"ERA5Client initialized in {self.data_mode.upper()} mode")

    def get_snapshot(
        self,
        storm_id: str,
        timestamp: str,
        center_lat: float = 15.0,
        center_lon: float = 85.0,
    ) -> EnvironmentalSnapshot:
        """Fetch environmental data snapshot."""
        if self.data_mode == "synthetic":
            return self._generator.generate_snapshot(
                storm_id=storm_id,
                timestamp=timestamp,
                center_lat=center_lat,
                center_lon=center_lon,
            )
        else:
            raise NotImplementedError(
                "Real ERA5/CDS API not yet implemented. "
                "Use data_mode='synthetic' for prototyping."
            )

    def get_snapshot_sequence(
        self,
        storm_id: str,
        timestamps: list[str],
        centers: list[tuple[float, float]],
    ) -> list[EnvironmentalSnapshot]:
        """Fetch temporal sequence of environmental snapshots."""
        return [
            self.get_snapshot(storm_id, ts, lat, lon)
            for ts, (lat, lon) in zip(timestamps, centers)
        ]
