"""
Inference Contract — Stage 13 output.

The `run_inference()` function is the single entry point for Member B's
backend. It produces a JSON object matching the agreed schema exactly.
Every call is logged to MLflow.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── JSON Schema for output validation ──
OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "storm_id", "timestamp", "detection", "structure",
        "current_state", "forecast", "ri", "uncertainty", "xai",
    ],
    "properties": {
        "storm_id": {"type": "string"},
        "timestamp": {"type": "string"},
        "detection": {
            "type": "object",
            "required": ["present", "confidence", "center", "bbox", "mask_url"],
            "properties": {
                "present": {"type": "boolean"},
                "confidence": {"type": "number"},
                "center": {
                    "type": "object",
                    "required": ["lat", "lon"],
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                    },
                },
                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "mask_url": {"type": "string"},
            },
        },
        "structure": {
            "type": "object",
            "required": [
                "eye_present", "eye_confidence", "eye_center",
                "eye_diameter_km", "symmetry", "circularity",
            ],
            "properties": {
                "eye_present": {"type": "boolean"},
                "eye_confidence": {"type": "number"},
                "eye_center": {
                    "type": "object",
                    "required": ["lat", "lon"],
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                    },
                },
                "eye_diameter_km": {"type": "number"},
                "symmetry": {"type": "number"},
                "circularity": {"type": "number"},
            },
        },
        "current_state": {
            "type": "object",
            "required": ["category", "stage", "pattern"],
            "properties": {
                "category": {"type": "string"},
                "stage": {"type": "string"},
                "pattern": {"type": "string"},
            },
        },
        "forecast": {
            "type": "object",
            "required": ["track", "intensity"],
            "properties": {
                "track": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["t", "lat", "lon", "cone_radius_km"],
                    },
                },
                "intensity": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["t", "wind_kt", "interval"],
                    },
                },
            },
        },
        "ri": {
            "type": "object",
            "required": ["probability", "confidence"],
            "properties": {
                "probability": {"type": "number"},
                "confidence": {"type": "number"},
            },
        },
        "uncertainty": {
            "type": "object",
            "required": ["novelty_score", "human_review_flag"],
            "properties": {
                "novelty_score": {"type": "number"},
                "human_review_flag": {"type": "boolean"},
            },
        },
        "xai": {
            "type": "object",
            "required": ["gradcam_url", "explanation_tags"],
            "properties": {
                "gradcam_url": {"type": "string"},
                "explanation_tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def validate_output(output: dict) -> bool:
    """
    Validate that the output dict matches the contract schema.

    Args:
        output: Prediction dict to validate.

    Returns:
        True if valid.

    Raises:
        jsonschema.ValidationError if invalid.
    """
    import jsonschema
    jsonschema.validate(output, OUTPUT_SCHEMA)
    return True


def run_inference(
    storm_id: str,
    timestamp: str,
    config_dir: str = "configs",
    log_to_mlflow: bool = True,
    device: Optional[str] = None,
) -> dict:
    """
    Run the complete cyclone prediction pipeline.

    This is the primary entry point for Member B's backend.
    Every call produces a JSON-serializable dict matching the
    agreed output contract schema.

    Args:
        storm_id: Storm identifier (IBTrACS SID, e.g., "2020121N10087").
        timestamp: ISO8601 timestamp for the prediction.
        config_dir: Path to configuration directory.
        log_to_mlflow: Whether to log the run to MLflow.
        device: Torch device ("cuda", "cpu", or "auto").

    Returns:
        Prediction dict conforming to the output contract.
        See OUTPUT_SCHEMA for the exact structure.

    Example:
        >>> result = run_inference("2020121N10087", "2020-05-17T06:00:00Z")
        >>> print(result["detection"]["center"])
        {'lat': 12.5, 'lon': 85.8}
        >>> print(result["current_state"]["category"])
        'VSCS'
    """
    start_time = time.time()

    # Initialize pipeline
    from cyclone_ai.pipeline import CyclonePipeline
    pipeline = CyclonePipeline(config_dir=config_dir, device=device)

    # Run pipeline
    result = pipeline.run(storm_id=storm_id, timestamp=timestamp)

    # Validate output
    try:
        validate_output(result)
        logger.info("Output validation: PASSED ✓")
    except Exception as e:
        logger.error(f"Output validation FAILED: {e}")
        raise

    elapsed = time.time() - start_time

    # Log to MLflow
    if log_to_mlflow:
        _log_mlflow(storm_id, timestamp, result, elapsed, config_dir)

    logger.info(
        f"Inference complete: storm={storm_id}, time={timestamp}, "
        f"category={result['current_state']['category']}, "
        f"elapsed={elapsed:.2f}s"
    )

    return result


def _log_mlflow(
    storm_id: str,
    timestamp: str,
    result: dict,
    elapsed: float,
    config_dir: str,
):
    """Log inference run to MLflow."""
    try:
        import mlflow
        import yaml

        # Load MLflow config
        inference_config_path = Path(config_dir) / "inference.yaml"
        if inference_config_path.exists():
            with open(inference_config_path) as f:
                config = yaml.safe_load(f) or {}
            mlflow_config = config.get("mlflow", {})
        else:
            mlflow_config = {}

        tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")
        experiment_name = mlflow_config.get("experiment_name", "cyclone-prediction")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{storm_id}_{timestamp[:10]}"):
            # Log parameters
            mlflow.log_param("storm_id", storm_id)
            mlflow.log_param("timestamp", timestamp)

            # Log metrics
            mlflow.log_metric("inference_time_s", elapsed)
            mlflow.log_metric("detection_confidence", result["detection"]["confidence"])
            mlflow.log_metric("ri_probability", result["ri"]["probability"])
            mlflow.log_metric("novelty_score", result["uncertainty"]["novelty_score"])

            if result["structure"]["eye_present"]:
                mlflow.log_metric("eye_confidence", result["structure"]["eye_confidence"])
                mlflow.log_metric("symmetry", result["structure"]["symmetry"])

            # Log category as tag
            mlflow.set_tag("category", result["current_state"]["category"])
            mlflow.set_tag("stage", result["current_state"]["stage"])
            mlflow.set_tag("human_review", str(result["uncertainty"]["human_review_flag"]))

            # Log result JSON as artifact
            result_json = json.dumps(result, indent=2)
            result_path = Path("outputs") / f"{storm_id}_{timestamp[:10]}_result.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(result_json)
            mlflow.log_artifact(str(result_path))

        logger.info(f"MLflow run logged: {experiment_name}")

    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")
