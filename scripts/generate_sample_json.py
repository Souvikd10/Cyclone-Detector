"""
Generate sample JSON outputs for Member B (backend) and Member C (frontend).

Produces 3 realistic sample JSONs for Amphan, Fani, and Mocha with
plausible values matching the output contract schema.
"""

import json
import sys
from pathlib import Path


def generate_amphan_sample() -> dict:
    """Super Cyclonic Storm Amphan (2020) — peak intensity sample."""
    return {
        "storm_id": "2020121N10087",
        "timestamp": "2020-05-18T12:00:00Z",
        "detection": {
            "present": True,
            "confidence": 0.97,
            "center": {"lat": 16.0, "lon": 85.2},
            "bbox": [13.0, 82.2, 19.0, 88.2],
            "mask_url": "minio://cyclone-artifacts/inference/2020121N10087/mask.png",
        },
        "structure": {
            "eye_present": True,
            "eye_confidence": 0.92,
            "eye_center": {"lat": 16.05, "lon": 85.18},
            "eye_diameter_km": 45.0,
            "symmetry": 0.85,
            "circularity": 0.82,
        },
        "current_state": {
            "category": "SuCS",
            "stage": "mature",
            "pattern": "eye_pattern",
        },
        "forecast": {
            "track": [
                {"t": "+6h", "lat": 16.8, "lon": 85.5, "cone_radius_km": 55},
                {"t": "+12h", "lat": 17.5, "lon": 86.0, "cone_radius_km": 90},
                {"t": "+24h", "lat": 19.0, "lon": 87.0, "cone_radius_km": 145},
                {"t": "+48h", "lat": 21.5, "lon": 88.5, "cone_radius_km": 260},
                {"t": "+72h", "lat": 23.5, "lon": 89.5, "cone_radius_km": 390},
            ],
            "intensity": [
                {"t": "+6h", "wind_kt": 125, "interval": [115, 135]},
                {"t": "+12h", "wind_kt": 115, "interval": [100, 130]},
                {"t": "+24h", "wind_kt": 90, "interval": [70, 110]},
                {"t": "+48h", "wind_kt": 55, "interval": [35, 75]},
                {"t": "+72h", "wind_kt": 30, "interval": [15, 45]},
            ],
        },
        "ri": {
            "probability": 0.12,
            "confidence": 0.88,
        },
        "uncertainty": {
            "novelty_score": 1.2,
            "human_review_flag": False,
        },
        "xai": {
            "gradcam_url": "minio://cyclone-artifacts/inference/2020121N10087/gradcam.png",
            "explanation_tags": [
                "well-defined eye",
                "strong eyewall",
                "symmetric convection",
                "peak intensity",
            ],
        },
    }


def generate_fani_sample() -> dict:
    """Extremely Severe Cyclonic Storm Fani (2019) — intensifying phase."""
    return {
        "storm_id": "2019121N08085",
        "timestamp": "2019-04-30T12:00:00Z",
        "detection": {
            "present": True,
            "confidence": 0.94,
            "center": {"lat": 13.5, "lon": 83.6},
            "bbox": [10.5, 80.6, 16.5, 86.6],
            "mask_url": "minio://cyclone-artifacts/inference/2019121N08085/mask.png",
        },
        "structure": {
            "eye_present": True,
            "eye_confidence": 0.78,
            "eye_center": {"lat": 13.52, "lon": 83.58},
            "eye_diameter_km": 35.0,
            "symmetry": 0.72,
            "circularity": 0.68,
        },
        "current_state": {
            "category": "ESCS",
            "stage": "intensifying",
            "pattern": "CDO",
        },
        "forecast": {
            "track": [
                {"t": "+6h", "lat": 14.2, "lon": 83.7, "cone_radius_km": 50},
                {"t": "+12h", "lat": 14.8, "lon": 83.9, "cone_radius_km": 85},
                {"t": "+24h", "lat": 16.5, "lon": 84.5, "cone_radius_km": 140},
                {"t": "+48h", "lat": 19.5, "lon": 85.5, "cone_radius_km": 250},
                {"t": "+72h", "lat": 22.0, "lon": 87.0, "cone_radius_km": 375},
            ],
            "intensity": [
                {"t": "+6h", "wind_kt": 100, "interval": [85, 115]},
                {"t": "+12h", "wind_kt": 115, "interval": [95, 130]},
                {"t": "+24h", "wind_kt": 125, "interval": [105, 140]},
                {"t": "+48h", "wind_kt": 85, "interval": [60, 110]},
                {"t": "+72h", "wind_kt": 45, "interval": [25, 65]},
            ],
        },
        "ri": {
            "probability": 0.68,
            "confidence": 0.75,
        },
        "uncertainty": {
            "novelty_score": 0.9,
            "human_review_flag": False,
        },
        "xai": {
            "gradcam_url": "minio://cyclone-artifacts/inference/2019121N08085/gradcam.png",
            "explanation_tags": [
                "developing eye",
                "rapid intensification likely",
                "organized convection",
                "low wind shear environment",
            ],
        },
    }


def generate_mocha_sample() -> dict:
    """Extremely Severe Cyclonic Storm Mocha (2023) — near peak."""
    return {
        "storm_id": "2023126N10091",
        "timestamp": "2023-05-13T00:00:00Z",
        "detection": {
            "present": True,
            "confidence": 0.98,
            "center": {"lat": 17.5, "lon": 89.5},
            "bbox": [14.5, 86.5, 20.5, 92.5],
            "mask_url": "minio://cyclone-artifacts/inference/2023126N10091/mask.png",
        },
        "structure": {
            "eye_present": True,
            "eye_confidence": 0.95,
            "eye_center": {"lat": 17.48, "lon": 89.52},
            "eye_diameter_km": 30.0,
            "symmetry": 0.90,
            "circularity": 0.88,
        },
        "current_state": {
            "category": "SuCS",
            "stage": "mature",
            "pattern": "pinhole_eye",
        },
        "forecast": {
            "track": [
                {"t": "+6h", "lat": 18.0, "lon": 89.8, "cone_radius_km": 45},
                {"t": "+12h", "lat": 18.5, "lon": 90.0, "cone_radius_km": 80},
                {"t": "+24h", "lat": 19.5, "lon": 91.0, "cone_radius_km": 135},
                {"t": "+48h", "lat": 21.0, "lon": 92.5, "cone_radius_km": 245},
                {"t": "+72h", "lat": 22.5, "lon": 93.5, "cone_radius_km": 370},
            ],
            "intensity": [
                {"t": "+6h", "wind_kt": 130, "interval": [120, 140]},
                {"t": "+12h", "wind_kt": 120, "interval": [105, 135]},
                {"t": "+24h", "wind_kt": 90, "interval": [70, 110]},
                {"t": "+48h", "wind_kt": 50, "interval": [30, 70]},
                {"t": "+72h", "wind_kt": 25, "interval": [10, 40]},
            ],
        },
        "ri": {
            "probability": 0.08,
            "confidence": 0.92,
        },
        "uncertainty": {
            "novelty_score": 1.8,
            "human_review_flag": False,
        },
        "xai": {
            "gradcam_url": "minio://cyclone-artifacts/inference/2023126N10091/gradcam.png",
            "explanation_tags": [
                "well-defined eye",
                "pinhole eye structure",
                "strong eyewall",
                "symmetric convection",
                "approaching landfall",
            ],
        },
    }


def main():
    """Generate and save all sample JSONs."""
    output_dir = Path("sample_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "amphan_sample.json": generate_amphan_sample(),
        "fani_sample.json": generate_fani_sample(),
        "mocha_sample.json": generate_mocha_sample(),
    }

    for filename, data in samples.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Generated {filepath}")

    # Also validate all samples against the schema
    try:
        from cyclone_ai.inference import validate_output
        for filename, data in samples.items():
            validate_output(data)
            print(f"[OK] Validated {filename}")
        print("\nAll samples validated successfully!")
    except ImportError:
        print("\n[INFO] Skipped deep validation (cyclone_ai dependencies optional for script generation)")
    except Exception as e:
        print(f"\n[ERROR] Validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
