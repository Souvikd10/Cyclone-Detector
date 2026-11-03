"""
Test suite for the SIH-2K26 Cyclone Prediction API.

Uses httpx.AsyncClient against the FastAPI test app.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from uuid import UUID

from app.main import app


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_root(client: AsyncClient):
    """Health check returns API info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_list_storms_catalog(client: AsyncClient):
    """GET /storms returns active and historical storms."""
    response = await client.get("/storms")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 5
    assert len(data["storms"]) >= 5



@pytest.mark.asyncio
async def test_seed_then_list(client: AsyncClient):
    """POST /seed then GET /storms returns seeded storms."""
    # Seed
    seed_resp = await client.post("/seed")
    assert seed_resp.status_code == 200
    seed_data = seed_resp.json()
    assert seed_data["storms_created"] == 12
    assert seed_data["predictions_created"] == 12 * 5

    # List
    list_resp = await client.get("/storms")
    assert list_resp.status_code == 200
    storms = list_resp.json()
    assert storms["count"] == 12

    # Filter active
    active_resp = await client.get("/storms?status=active")
    assert active_resp.status_code == 200
    active_data = active_resp.json()
    assert active_data["count"] == 3



@pytest.mark.asyncio
async def test_latest_prediction(client: AsyncClient):
    """GET /storms/{id}/latest returns the most recent prediction."""
    # Seed first
    seed_resp = await client.post("/seed")
    storm_id = seed_resp.json()["storm_ids"][0]

    # Get latest
    resp = await client.get(f"/storms/{storm_id}/latest")
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["storm_id"] == storm_id
    assert "detection" in pred
    assert "forecast" in pred
    assert "impact" in pred


@pytest.mark.asyncio
async def test_prediction_history(client: AsyncClient):
    """GET /storms/{id}/history returns time series of predictions."""
    seed_resp = await client.post("/seed")
    storm_id = seed_resp.json()["storm_ids"][0]

    resp = await client.get(f"/storms/{storm_id}/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 5
    assert len(data["predictions"]) == 5


@pytest.mark.asyncio
async def test_impact_endpoint(client: AsyncClient):
    """GET /storms/{id}/impact returns Stage 14 impact data."""
    seed_resp = await client.post("/seed")
    storm_id = seed_resp.json()["storm_ids"][0]

    resp = await client.get(f"/storms/{storm_id}/impact")
    assert resp.status_code == 200
    impact = resp.json()
    assert impact["storm_id"] == storm_id
    assert "landfall_location" in impact
    assert "impact_zone" in impact


@pytest.mark.asyncio
async def test_alert_json(client: AsyncClient):
    """GET /alerts/{storm_id} returns CAP alert as JSON."""
    seed_resp = await client.post("/seed")
    storm_id = seed_resp.json()["storm_ids"][0]

    resp = await client.get(f"/alerts/{storm_id}")
    assert resp.status_code == 200
    alert = resp.json()
    assert "info" in alert
    assert "cap_xml" in alert
    assert alert["info"]["status"] == "Test"  # Always prototype
    assert "<?xml" in alert["cap_xml"]


@pytest.mark.asyncio
async def test_alert_xml(client: AsyncClient):
    """GET /alerts/{storm_id}?format=xml returns raw CAP XML."""
    seed_resp = await client.post("/seed")
    storm_id = seed_resp.json()["storm_ids"][0]

    resp = await client.get(f"/alerts/{storm_id}?format=xml")
    assert resp.status_code == 200
    assert "xml" in resp.headers.get("content-type", "")
    assert b"<alert" in resp.content


@pytest.mark.asyncio
async def test_feedback_accept(client: AsyncClient):
    """POST /storms/{id}/feedback with action=accept."""
    seed_resp = await client.post("/seed")
    storm_id = seed_resp.json()["storm_ids"][0]

    # Get a prediction ID
    pred_resp = await client.get(f"/storms/{storm_id}/latest")
    prediction_id = pred_resp.json()["id"]

    # Submit feedback
    resp = await client.post(
        f"/storms/{storm_id}/feedback",
        json={
            "prediction_id": prediction_id,
            "action": "accept",
            "forecaster_note": "Prediction looks accurate",
            "forecaster_id": "test-forecaster",
        },
    )
    assert resp.status_code == 201
    fb = resp.json()
    assert fb["action"] == "accept"
    assert fb["storm_id"] == storm_id


@pytest.mark.asyncio
async def test_feedback_override(client: AsyncClient):
    """POST /storms/{id}/feedback with action=override."""
    seed_resp = await client.post("/seed")
    storm_id = seed_resp.json()["storm_ids"][0]

    pred_resp = await client.get(f"/storms/{storm_id}/latest")
    prediction_id = pred_resp.json()["id"]

    resp = await client.post(
        f"/storms/{storm_id}/feedback",
        json={
            "prediction_id": prediction_id,
            "action": "override",
            "corrected_value": {"max_wind_knots": 90, "category": "VSCS"},
            "forecaster_note": "Wind speed underestimated based on reconnaissance",
            "forecaster_id": "test-forecaster",
        },
    )
    assert resp.status_code == 201
    fb = resp.json()
    assert fb["action"] == "override"
    assert fb["corrected_value"]["max_wind_knots"] == 90


@pytest.mark.asyncio
async def test_storm_not_found(client: AsyncClient):
    """404 for non-existent storm ID."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    resp = await client.get(f"/storms/{fake_id}/latest")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_invalid_status_filter(client: AsyncClient):
    """400 for invalid status filter."""
    resp = await client.get("/storms?status=bogus")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_ingest_member_a_prediction(client: AsyncClient):
    """POST /storms/ingest ingests a Member A Contract JSON."""
    contract = {
        "storm_id": "amphan",
        "timestamp": "2020-05-18T12:00:00Z",
        "detection": {
            "present": True,
            "confidence": 0.98,
            "center": {"lat": 16.0, "lon": 85.2},
            "bbox": [13.0, 82.2, 19.0, 88.2],
            "mask_url": "minio://cyclone-artifacts/inference/amphan/mask.png",
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
            ],
            "intensity": [
                {"t": "+6h", "wind_kt": 125, "interval": [115, 135]},
                {"t": "+12h", "wind_kt": 115, "interval": [100, 130]},
                {"t": "+24h", "wind_kt": 90, "interval": [70, 110]},
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
            "gradcam_url": "minio://cyclone-artifacts/inference/amphan/gradcam.png",
            "explanation_tags": ["well-defined eye", "strong eyewall"],
        },
    }
    resp = await client.post("/storms/ingest", json=contract)
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["current_state"]["category"] == "SuCS"
    assert pred["detection"]["detected"] is True
    assert "impact" in pred
    assert "forecast" in pred
    assert len(pred["forecast"]["track"]) == 3


@pytest.mark.asyncio
async def test_trigger_prediction(client: AsyncClient):
    """POST /storms/{storm_id}/predict generates or runs prediction for a storm."""
    seed_resp = await client.post("/seed")
    storm_id = seed_resp.json()["storm_ids"][0]

    resp = await client.post(f"/storms/{storm_id}/predict")
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["storm_id"] == storm_id
    assert "current_state" in pred
    assert "forecast" in pred
    assert "impact" in pred
