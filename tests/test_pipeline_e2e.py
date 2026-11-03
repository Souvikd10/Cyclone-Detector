"""
End-to-end pipeline test with real model heads.
"""

import json
import pytest


class TestPipelineE2E:
    """End-to-end pipeline integration test."""

    def test_full_pipeline_produces_valid_contract(self):
        """Run the complete pipeline and validate contract compliance."""
        from cyclone_ai.inference import run_inference, validate_output

        result = run_inference(
            storm_id="2020121N10087",
            timestamp="2020-05-17T06:00:00Z",
            config_dir="configs",
            log_to_mlflow=False,
            device="cpu",
        )

        # Must validate against contract schema
        assert validate_output(result) is True

        # Check all top-level keys
        assert result["storm_id"] == "2020121N10087"
        assert result["timestamp"] == "2020-05-17T06:00:00Z"
        assert "detection" in result
        assert "structure" in result
        assert "current_state" in result
        assert "forecast" in result
        assert "ri" in result
        assert "uncertainty" in result
        assert "xai" in result

    def test_detection_output(self):
        """Verify detection output structure."""
        from cyclone_ai.inference import run_inference

        result = run_inference(
            "2020121N10087", "2020-05-17T06:00:00Z",
            log_to_mlflow=False, device="cpu",
        )

        det = result["detection"]
        assert isinstance(det["present"], bool)
        assert 0 <= det["confidence"] <= 1
        assert "lat" in det["center"]
        assert "lon" in det["center"]
        assert len(det["bbox"]) == 4
        assert isinstance(det["mask_url"], str)

    def test_forecast_horizons(self):
        """Verify forecast has correct number of horizons."""
        from cyclone_ai.inference import run_inference

        result = run_inference(
            "2020121N10087", "2020-05-17T06:00:00Z",
            log_to_mlflow=False, device="cpu",
        )

        track = result["forecast"]["track"]
        intensity = result["forecast"]["intensity"]

        assert len(track) == 5
        assert len(intensity) == 5

        # Check horizons
        expected_horizons = ["+6h", "+12h", "+24h", "+48h", "+72h"]
        for i, h in enumerate(expected_horizons):
            assert track[i]["t"] == h
            assert intensity[i]["t"] == h

    def test_xai_output(self):
        """Verify XAI output has gradcam URL and tags."""
        from cyclone_ai.inference import run_inference

        result = run_inference(
            "2020121N10087", "2020-05-17T06:00:00Z",
            log_to_mlflow=False, device="cpu",
        )

        xai = result["xai"]
        assert isinstance(xai["gradcam_url"], str)
        assert len(xai["explanation_tags"]) > 0
        assert all(isinstance(tag, str) for tag in xai["explanation_tags"])

    def test_result_is_json_serializable(self):
        """Ensure the entire result can be serialized to JSON."""
        from cyclone_ai.inference import run_inference

        result = run_inference(
            "2020121N10087", "2020-05-17T06:00:00Z",
            log_to_mlflow=False, device="cpu",
        )

        json_str = json.dumps(result, indent=2)
        parsed = json.loads(json_str)
        assert parsed["storm_id"] == "2020121N10087"
