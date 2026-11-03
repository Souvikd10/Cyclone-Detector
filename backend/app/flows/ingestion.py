"""
Prefect Flows for the SIH-2K26 Cyclone Prediction Pipeline.

Flows:
1. ingestion_flow — Periodic data ingestion (calls Member A's pipeline)
2. retrain_check_flow — Monitors feedback override rate and triggers retraining
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# NOTE: Prefect imports are deferred to avoid import errors when Prefect is not installed.
# These flows are meant to be registered with the Prefect server and run as scheduled deployments.

try:
    from prefect import flow, task
    from prefect.tasks import task_input_hash
    from datetime import timedelta

    @task(
        name="fetch_satellite_data",
        retries=3,
        retry_delay_seconds=60,
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
    )
    def fetch_satellite_data(storm_id: str) -> dict:
        """
        Fetch latest satellite imagery for a storm.
        In production, this calls Member A's data ingestion pipeline.
        """
        logger.info(f"[STUB] Fetching satellite data for storm {storm_id}")
        # TODO: Wire to Member A's data source (INSAT-3D, Sentinel, etc.)
        return {
            "storm_id": storm_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "mock",
            "bands": ["IR", "VIS", "WV"],
        }

    @task(name="run_inference")
    def run_inference(satellite_data: dict) -> dict:
        """
        Run Member A's AI inference pipeline.
        Calls cyclone_ai if available, or formats via ai_bridge.
        """
        storm_id = satellite_data["storm_id"]
        timestamp = satellite_data.get("timestamp")
        logger.info(f"Running Member A inference pipeline for storm {storm_id}")
        
        # Try direct Member A pipeline execution
        try:
            from cyclone_ai.inference import run_inference as run_cyclone_ai
            return run_cyclone_ai(storm_id=storm_id, timestamp=timestamp, log_to_mlflow=False)
        except Exception as e:
            logger.info(f"Direct cyclone_ai inference fallback ({e}). Using mock/sample generator.")
        
        from app.core.mock_data import MOCK_STORMS, generate_mock_prediction
        import uuid

        template = MOCK_STORMS[0] if MOCK_STORMS else {
            "name": "Detected Storm",
            "basin": "NIO",
            "category": "CS",
            "base_lat": 14.0,
            "base_lon": 85.0,
            "wind": 55,
            "pressure": 994,
            "target_lat": 18.0,
            "target_lon": 84.0,
        }
        try:
            s_uuid = uuid.UUID(storm_id)
        except ValueError:
            s_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, storm_id)

        prediction = generate_mock_prediction(
            storm_id=s_uuid,
            storm_template=template,
        )
        return prediction

    @task(name="store_prediction")
    def store_prediction(prediction: dict) -> str:
        """
        Store prediction to the database.
        In production, this writes to PostgreSQL via SQLAlchemy.
        """
        prediction_id = str(prediction.get("id", "unknown"))
        logger.info(f"[STUB] Storing prediction {prediction_id}")
        # TODO: Use synchronous DB session to write prediction to DB
        return prediction_id

    @task(name="broadcast_update")
    def broadcast_update(storm_id: str, prediction: dict):
        """
        Broadcast new prediction via WebSocket to connected dashboards.
        """
        logger.info(f"[STUB] Broadcasting update for storm {storm_id}")
        # TODO: Use the WebSocket manager to broadcast
        # In practice, this would be an async call or HTTP request to the API

    @flow(
        name="cyclone-ingestion",
        description="Periodic ingestion flow: fetch satellite data → run inference → store → broadcast",
        log_prints=True,
    )
    def ingestion_flow(storm_ids: list[str] | None = None):
        """
        Main ingestion flow — runs every 6 hours.

        For each active storm:
        1. Fetch latest satellite imagery
        2. Run AI inference (Member A's pipeline)
        3. Store prediction to database
        4. Broadcast update via WebSocket
        """
        if storm_ids is None:
            # TODO: Query DB for active storm IDs
            logger.info("[STUB] No storm IDs provided, using mock data")
            storm_ids = ["mock-storm-001"]

        logger.info(f"Starting ingestion for {len(storm_ids)} storms")

        for storm_id in storm_ids:
            logger.info(f"Processing storm {storm_id}")
            satellite_data = fetch_satellite_data(storm_id)
            prediction = run_inference(satellite_data)
            pred_id = store_prediction(prediction)
            broadcast_update(storm_id, prediction)
            logger.info(f"Storm {storm_id}: prediction {pred_id} complete")

        logger.info(f"Ingestion complete for {len(storm_ids)} storms")

    @flow(
        name="retrain-check",
        description="Check feedback override rates and trigger retraining if threshold exceeded",
        log_prints=True,
    )
    def retrain_check_flow():
        """
        Retrain check flow — runs daily.

        Checks the feedback override rate across all active storms.
        If the rate exceeds the threshold, logs a retrain trigger.
        """
        logger.info("Running retrain check...")
        # TODO: Query feedback table for override rates
        # TODO: If threshold exceeded, trigger retraining via Member A's pipeline
        logger.info("[STUB] Retrain check complete — no action taken")

    logger.info("Prefect flows registered successfully")

except ImportError:
    logger.warning(
        "Prefect not installed — flow definitions skipped. "
        "Install with: pip install prefect"
    )

    # Provide stub functions so imports don't break
    def ingestion_flow(storm_ids=None):
        logger.warning("Prefect not available — ingestion_flow is a stub")

    def retrain_check_flow():
        logger.warning("Prefect not available — retrain_check_flow is a stub")
