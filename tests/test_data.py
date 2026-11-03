"""
Tests for data acquisition and preprocessing (Stages 1–2).
"""

import numpy as np
import pytest


class TestSyntheticINSATGenerator:
    """Test synthetic satellite data generation."""

    def test_generator_creates_scene(self):
        from cyclone_ai.data.mosdac_client import SyntheticINSATGenerator

        gen = SyntheticINSATGenerator(image_size=64, seed=42)
        scene = gen.generate_scene(
            storm_id="TEST_001",
            timestamp="2020-05-17T06:00:00Z",
            center_lat=15.0,
            center_lon=85.0,
            intensity_factor=1.0,
            has_eye=True,
        )

        assert scene.storm_id == "TEST_001"
        assert len(scene.products) == 4  # IR_TIR1, IR_TIR2, WV, VIS

        for name, product in scene.products.items():
            assert product.data is not None
            assert product.data.shape == (64, 64)
            assert product.data.dtype == np.float32

    def test_scene_to_numpy(self):
        from cyclone_ai.data.mosdac_client import SyntheticINSATGenerator

        gen = SyntheticINSATGenerator(image_size=32, seed=42)
        scene = gen.generate_scene(
            "TEST", "2020-01-01T00:00:00Z", 15.0, 85.0,
        )
        arr = scene.to_numpy()
        assert arr.shape == (4, 32, 32)
        assert arr.dtype == np.float32

    def test_eye_affects_output(self):
        from cyclone_ai.data.mosdac_client import SyntheticINSATGenerator

        gen = SyntheticINSATGenerator(image_size=64, seed=42)
        scene_no_eye = gen.generate_scene(
            "TEST", "2020-01-01T00:00:00Z", 15.0, 85.0, has_eye=False,
        )
        gen2 = SyntheticINSATGenerator(image_size=64, seed=42)
        scene_eye = gen2.generate_scene(
            "TEST", "2020-01-01T00:00:00Z", 15.0, 85.0, has_eye=True,
        )

        # With eye should have different values at center
        ir1_no_eye = scene_no_eye.products["IR_TIR1"].data
        ir1_eye = scene_eye.products["IR_TIR1"].data
        center = 32
        assert ir1_no_eye[center, center] != ir1_eye[center, center]


class TestMOSDACClient:
    """Test MOSDAC client in synthetic mode."""

    def test_synthetic_mode_works(self, tmp_path):
        import yaml

        config = {
            "data_mode": "synthetic",
            "mosdac": {"download_dir": str(tmp_path / "mosdac")},
            "synthetic": {"image_size": 32, "channels": ["IR_TIR1", "IR_TIR2", "WV", "VIS"]},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from cyclone_ai.data.mosdac_client import MOSDACClient

        client = MOSDACClient(config_path=str(config_path))
        scene = client.get_scene("TEST", "2020-01-01T00:00:00Z")
        assert len(scene.products) == 4

    def test_scene_sequence(self, tmp_path):
        import yaml

        config = {
            "data_mode": "synthetic",
            "mosdac": {"download_dir": str(tmp_path / "mosdac")},
            "synthetic": {"image_size": 32},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from cyclone_ai.data.mosdac_client import MOSDACClient

        client = MOSDACClient(config_path=str(config_path))
        scenes = client.get_scene_sequence(
            "TEST",
            ["2020-01-01T00:00:00Z", "2020-01-01T06:00:00Z"],
            [(15.0, 85.0), (15.5, 85.2)],
        )
        assert len(scenes) == 2


class TestIBTrACSClient:
    """Test IBTrACS client."""

    def test_synthetic_fallback(self, tmp_path):
        import yaml

        config = {
            "ibtracs": {
                "csv_url": "https://example.com/nonexistent.csv",
                "download_dir": str(tmp_path / "ibtracs"),
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from cyclone_ai.data.ibtracs_client import IBTrACSClient

        client = IBTrACSClient(config_path=str(config_path))
        # Download will fail and fall back to synthetic
        df = client.load()
        assert len(df) > 0

        # Should find Amphan
        track = client.get_storm("2020121N10087")
        assert track.name == "AMPHAN"
        assert len(track.points) > 0
        assert track.peak_wind_kt > 100

    def test_imd_classification(self):
        from cyclone_ai.data.ibtracs_client import IBTrACSClient

        assert IBTrACSClient._classify_wind(10) == "LOW"
        assert IBTrACSClient._classify_wind(25) == "D"
        assert IBTrACSClient._classify_wind(35) == "CS"
        assert IBTrACSClient._classify_wind(65) == "VSCS"
        assert IBTrACSClient._classify_wind(125) == "SuCS"


class TestERA5Client:
    """Test ERA5 synthetic data."""

    def test_snapshot_generation(self, tmp_path):
        import yaml

        config = {
            "data_mode": "synthetic",
            "era5": {},
            "synthetic": {"image_size": 32},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from cyclone_ai.data.era5_client import ERA5Client

        client = ERA5Client(config_path=str(config_path))
        snapshot = client.get_snapshot("TEST", "2020-01-01T00:00:00Z")

        assert "SST" in snapshot.fields
        assert "WIND_SHEAR" in snapshot.fields
        assert "MOISTURE" in snapshot.fields

        arr = snapshot.to_numpy()
        assert arr.shape == (3, 32, 32)

    def test_scalar_features(self, tmp_path):
        import yaml

        config = {
            "data_mode": "synthetic",
            "era5": {},
            "synthetic": {"image_size": 64},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from cyclone_ai.data.era5_client import ERA5Client

        client = ERA5Client(config_path=str(config_path))
        snapshot = client.get_snapshot("TEST", "2020-01-01T00:00:00Z", 15.0, 85.0)
        scalars = snapshot.get_scalar_features(15.0, 85.0)

        assert "SST_mean" in scalars
        assert "WIND_SHEAR_mean" in scalars
        assert 25.0 < scalars["SST_mean"] < 35.0  # realistic SST range


class TestPreprocessing:
    """Test preprocessing utilities."""

    def test_time_alignment(self):
        from cyclone_ai.data.preprocessing import TimeAligner

        aligner = TimeAligner(resolution_minutes=30)
        # 14:17 → should snap to 14:30
        snapped = aligner.snap_timestamp("2020-05-17T14:17:00+00:00")
        assert "14:30" in snapped

    def test_normalizer(self):
        from cyclone_ai.data.preprocessing import Normalizer

        norm = Normalizer()
        data = np.random.randn(3, 64, 64).astype(np.float32) * 10 + 5
        channels = ["A", "B", "C"]

        result = norm.fit_transform(data, channels)
        # After z-score normalization, mean should be ~0 and std ~1
        for i in range(3):
            assert abs(result[i].mean()) < 0.5
            assert abs(result[i].std() - 1.0) < 0.5

    def test_missing_data_handler(self):
        from cyclone_ai.data.preprocessing import MissingDataHandler

        handler = MissingDataHandler(max_missing_frac=0.3)
        data = np.random.randn(2, 32, 32).astype(np.float32)

        # Add some NaN gaps
        data[0, 10:12, 10:12] = np.nan  # small gap
        cleaned, quality = handler.handle(data, ["ch1", "ch2"])

        assert not np.any(np.isnan(cleaned))
        assert quality["ch1"]["interpolated"] is True
        assert quality["ch2"]["missing_fraction"] == 0.0


class TestStormCropper:
    """Test storm-centered cropping."""

    def test_crop_shape(self):
        from cyclone_ai.data.storm_cropper import StormCropper

        cropper = StormCropper(crop_size=128, field_of_view_deg=10.0)
        data = np.random.randn(7, 512, 512).astype(np.float32)

        patch = cropper.crop(data, center_lat=15.0, center_lon=85.0)
        assert patch.shape == (7, 128, 128)

    def test_crop_with_metadata(self):
        from cyclone_ai.data.storm_cropper import StormCropper

        cropper = StormCropper(crop_size=64, field_of_view_deg=10.0)
        data = np.random.randn(4, 256, 256).astype(np.float32)

        result = cropper.crop_with_metadata(data, center_lat=15.0, center_lon=85.0)
        assert "data" in result
        assert "center_lat" in result
        assert result["data"].shape[0] == 4
