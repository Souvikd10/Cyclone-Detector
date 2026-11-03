"""
Tests for the inference contract — validates output JSON schema.
"""

import json
from pathlib import Path

import pytest


class TestOutputSchema:
    """Validate that outputs match the agreed contract schema."""

    def test_sample_amphan_validates(self):
        from cyclone_ai.inference import validate_output

        sample_path = Path("sample_outputs/amphan_sample.json")
        with open(sample_path) as f:
            data = json.load(f)

        assert validate_output(data) is True

    def test_sample_fani_validates(self):
        from cyclone_ai.inference import validate_output

        sample_path = Path("sample_outputs/fani_sample.json")
        with open(sample_path) as f:
            data = json.load(f)

        assert validate_output(data) is True

    def test_sample_mocha_validates(self):
        from cyclone_ai.inference import validate_output

        sample_path = Path("sample_outputs/mocha_sample.json")
        with open(sample_path) as f:
            data = json.load(f)

        assert validate_output(data) is True

    def test_missing_field_fails(self):
        from cyclone_ai.inference import validate_output
        import jsonschema

        incomplete = {
            "storm_id": "TEST",
            "timestamp": "2020-01-01T00:00:00Z",
            # Missing required fields
        }

        with pytest.raises(jsonschema.ValidationError):
            validate_output(incomplete)

    def test_wrong_type_fails(self):
        from cyclone_ai.inference import validate_output
        import jsonschema

        bad_data = {
            "storm_id": "TEST",
            "timestamp": "2020-01-01T00:00:00Z",
            "detection": {
                "present": "yes",  # should be boolean
                "confidence": 0.9,
                "center": {"lat": 0, "lon": 0},
                "bbox": [0, 0, 0, 0],
                "mask_url": "test",
            },
            "structure": {
                "eye_present": True, "eye_confidence": 0.0,
                "eye_center": {"lat": 0, "lon": 0},
                "eye_diameter_km": 0, "symmetry": 0, "circularity": 0,
            },
            "current_state": {"category": "CS", "stage": "mature", "pattern": "CDO"},
            "forecast": {
                "track": [{"t": "+6h", "lat": 0, "lon": 0, "cone_radius_km": 0}],
                "intensity": [{"t": "+6h", "wind_kt": 0, "interval": [0, 0]}],
            },
            "ri": {"probability": 0.0, "confidence": 0.0},
            "uncertainty": {"novelty_score": 0.0, "human_review_flag": False},
            "xai": {"gradcam_url": "test", "explanation_tags": []},
        }

        with pytest.raises(jsonschema.ValidationError):
            validate_output(bad_data)

    def test_all_required_fields_present(self):
        """Check that a minimal valid output passes."""
        from cyclone_ai.inference import validate_output

        minimal = {
            "storm_id": "TEST_001",
            "timestamp": "2024-01-01T00:00:00Z",
            "detection": {
                "present": False,
                "confidence": 0.1,
                "center": {"lat": 10.0, "lon": 80.0},
                "bbox": [8.0, 78.0, 12.0, 82.0],
                "mask_url": "minio://bucket/mask.png",
            },
            "structure": {
                "eye_present": False,
                "eye_confidence": 0.0,
                "eye_center": {"lat": 0.0, "lon": 0.0},
                "eye_diameter_km": 0.0,
                "symmetry": 0.3,
                "circularity": 0.2,
            },
            "current_state": {
                "category": "D",
                "stage": "forming",
                "pattern": "banding",
            },
            "forecast": {
                "track": [
                    {"t": "+6h", "lat": 10.2, "lon": 80.1, "cone_radius_km": 50},
                ],
                "intensity": [
                    {"t": "+6h", "wind_kt": 25, "interval": [15, 35]},
                ],
            },
            "ri": {"probability": 0.05, "confidence": 0.6},
            "uncertainty": {"novelty_score": 1.5, "human_review_flag": True},
            "xai": {
                "gradcam_url": "minio://bucket/gradcam.png",
                "explanation_tags": ["developing system"],
            },
        }

        assert validate_output(minimal) is True

    def test_bbox_must_have_four_elements(self):
        """BBox must be exactly 4 numbers."""
        from cyclone_ai.inference import validate_output
        import jsonschema

        data = {
            "storm_id": "TEST",
            "timestamp": "2020-01-01T00:00:00Z",
            "detection": {
                "present": True,
                "confidence": 0.9,
                "center": {"lat": 0, "lon": 0},
                "bbox": [0, 0, 0],  # Only 3 elements — should fail
                "mask_url": "test",
            },
            "structure": {
                "eye_present": True, "eye_confidence": 0.0,
                "eye_center": {"lat": 0, "lon": 0},
                "eye_diameter_km": 0, "symmetry": 0, "circularity": 0,
            },
            "current_state": {"category": "CS", "stage": "mature", "pattern": "CDO"},
            "forecast": {
                "track": [{"t": "+6h", "lat": 0, "lon": 0, "cone_radius_km": 0}],
                "intensity": [{"t": "+6h", "wind_kt": 0, "interval": [0, 0]}],
            },
            "ri": {"probability": 0.0, "confidence": 0.0},
            "uncertainty": {"novelty_score": 0.0, "human_review_flag": False},
            "xai": {"gradcam_url": "test", "explanation_tags": []},
        }

        with pytest.raises(jsonschema.ValidationError):
            validate_output(data)
