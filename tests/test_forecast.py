"""
Tests for forecast, RI, and physics correction (Stages 9–11).
"""

import numpy as np
import torch
import pytest


class TestForecastModel:
    """Test track/intensity forecast (Stage 9)."""

    def test_forward_shapes(self):
        from cyclone_ai.forecast.track_intensity import ForecastModel

        model = ForecastModel(temporal_dim=512, env_dim=8, model_dim=256, num_heads=4, num_layers=2)
        temporal = torch.randn(2, 512)
        env = torch.randn(2, 8)
        pos = torch.tensor([[15.0, 85.0], [13.5, 83.6]])

        out = model(temporal, env, current_position=pos)

        assert out["track_delta"].shape == (2, 5, 2)    # 5 horizons, lat+lon
        assert out["track_absolute"].shape == (2, 5, 2)
        assert out["cone_radius"].shape == (2, 5)
        assert out["wind_kt"].shape == (2, 5)
        assert out["interval_low"].shape == (2, 5)
        assert out["interval_high"].shape == (2, 5)

    def test_interval_ordering(self):
        from cyclone_ai.forecast.track_intensity import ForecastModel

        model = ForecastModel(temporal_dim=256, env_dim=4, model_dim=128, num_heads=4, num_layers=1)
        out = model(torch.randn(4, 256), torch.randn(4, 4))

        # interval_low <= wind_kt <= interval_high
        assert (out["interval_low"] <= out["wind_kt"] + 1e-5).all()
        assert (out["wind_kt"] <= out["interval_high"] + 1e-5).all()

    def test_non_negative_outputs(self):
        from cyclone_ai.forecast.track_intensity import ForecastModel

        model = ForecastModel(temporal_dim=128, env_dim=4, model_dim=64, num_heads=2, num_layers=1)
        out = model(torch.randn(2, 128), torch.randn(2, 4))

        assert (out["cone_radius"] >= 0).all()
        assert (out["wind_kt"] >= 0).all()

    def test_contract_output(self):
        from cyclone_ai.forecast.track_intensity import ForecastModel

        model = ForecastModel(temporal_dim=128, env_dim=4, model_dim=64, num_heads=2, num_layers=1)
        pos = torch.tensor([[15.0, 85.0]])
        out = model(torch.randn(1, 128), torch.randn(1, 4), current_position=pos)
        contract = model.to_contract(out)

        assert len(contract["track"]) == 5
        assert len(contract["intensity"]) == 5
        assert contract["track"][0]["t"] == "+6h"
        assert "lat" in contract["track"][0]
        assert "interval" in contract["intensity"][0]

    def test_haversine_loss(self):
        from cyclone_ai.forecast.track_intensity import HaversineLoss

        loss_fn = HaversineLoss()
        pred = torch.tensor([[[15.0, 85.0]]])
        target = torch.tensor([[[15.1, 85.1]]])
        loss = loss_fn(pred, target)

        # ~14 km for 0.1° change
        assert 10 < loss.item() < 20


class TestRIModel:
    """Test RI model (Stage 10)."""

    def test_feature_extraction(self):
        from cyclone_ai.forecast.ri_model import RIFeatureExtractor

        extractor = RIFeatureExtractor()
        features = extractor.extract(
            wind_kt=70, pressure_hpa=980, lat=15.0, lon=85.0,
            env_scalars={"SST_mean": 29.0, "WIND_SHEAR_mean": 8.0, "MOISTURE_mean": 70.0},
            structure={"symmetry": 0.7, "circularity": 0.6, "eye_present": True, "eye_diameter_km": 35},
        )

        assert features.shape == (len(extractor.FEATURE_NAMES),)
        assert features[0] == 70.0  # wind_kt

    def test_untrained_prediction(self):
        from cyclone_ai.forecast.ri_model import RIModel

        model = RIModel()
        features = np.random.randn(18).astype(np.float32)
        prob, conf = model.predict(features)

        # Untrained model returns defaults
        assert 0 <= prob <= 1
        assert 0 <= conf <= 1

    def test_contract_output(self):
        from cyclone_ai.forecast.ri_model import RIModel

        model = RIModel()
        features = np.random.randn(18).astype(np.float32)
        contract = model.to_contract(features)

        assert "probability" in contract
        assert "confidence" in contract
        assert 0 <= contract["probability"] <= 1


class TestPhysicsCorrection:
    """Test physics correction (Stage 11)."""

    def test_track_speed_clipping(self):
        from cyclone_ai.forecast.physics_correction import PhysicsCorrector

        corrector = PhysicsCorrector(max_track_speed_kt=30)

        forecast = {
            "track": [
                {"t": "+6h", "lat": 15.0, "lon": 85.0, "cone_radius_km": 50},
                {"t": "+12h", "lat": 25.0, "lon": 85.0, "cone_radius_km": 85},  # 10° jump!
            ],
            "intensity": [
                {"t": "+6h", "wind_kt": 80, "interval": [70, 90]},
                {"t": "+12h", "wind_kt": 75, "interval": [65, 85]},
            ],
        }

        corrected, _, corrections = corrector.correct(
            forecast, {"probability": 0.1, "confidence": 0.9},
            current_wind_kt=80, current_lat=14.0, current_lon=85.0,
        )

        # Should have been clipped
        assert len(corrections) > 0
        assert corrected["track"][1]["lat"] < 25.0  # no longer 10° jump

    def test_intensity_clipping(self):
        from cyclone_ai.forecast.physics_correction import PhysicsCorrector

        corrector = PhysicsCorrector(max_intensity_change_kt=40)

        forecast = {
            "track": [{"t": "+6h", "lat": 15, "lon": 85, "cone_radius_km": 50}],
            "intensity": [{"t": "+6h", "wind_kt": 200, "interval": [180, 220]}],  # +120kt!
        }

        corrected, _, corrections = corrector.correct(
            forecast, {"probability": 0.1, "confidence": 0.9},
            current_wind_kt=80, current_lat=15, current_lon=85,
        )

        assert corrected["intensity"][0]["wind_kt"] <= 120  # 80 + 40 max

    def test_imd_classification(self):
        from cyclone_ai.forecast.physics_correction import PhysicsCorrector

        assert PhysicsCorrector.classify_wind(10) == "LOW"
        assert PhysicsCorrector.classify_wind(30) == "DD"
        assert PhysicsCorrector.classify_wind(70) == "VSCS"
        assert PhysicsCorrector.classify_wind(130) == "SuCS"

    def test_sst_floor(self):
        from cyclone_ai.forecast.physics_correction import PhysicsCorrector

        corrector = PhysicsCorrector(sst_threshold_c=26.5)

        forecast = {
            "track": [{"t": "+6h", "lat": 15, "lon": 85, "cone_radius_km": 50}],
            "intensity": [{"t": "+6h", "wind_kt": 80, "interval": [70, 90]}],
        }

        corrected, _, corrections = corrector.correct(
            forecast, {"probability": 0.1, "confidence": 0.9},
            current_wind_kt=80, current_lat=15, current_lon=85,
            sst=24.0,  # Below threshold!
        )

        assert corrected["intensity"][0]["wind_kt"] <= 35
