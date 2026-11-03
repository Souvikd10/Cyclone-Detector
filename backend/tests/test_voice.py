"""
Test suite for Voice Agent Ephemeral Token Router:
  POST /voice/token
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_mint_voice_token_simulated(client: AsyncClient):
    """POST /voice/token returns 200 with simulated token when key is unconfigured."""
    response = await client.post("/voice/token", json={})
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert data["mode"] in ["simulated", "live"]
    assert "model" in data
    assert data["expires_in"] > 0
    assert "message" in data


@pytest.mark.asyncio
async def test_mint_voice_token_with_custom_model(client: AsyncClient):
    """POST /voice/token respects custom target model request."""
    response = await client.post(
        "/voice/token",
        json={"model": "models/gemini-2.0-flash-exp", "client_session_id": "test-session-123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "models/gemini-2.0-flash-exp"
