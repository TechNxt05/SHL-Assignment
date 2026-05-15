"""
Unit tests for SHL Recommender API schema compliance.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_chat_schema_clarify(client):
    """Test that initial chat returns a clarifying response with empty recommendations."""
    payload = {
        "messages": [{"role": "user", "content": "I need a test for my new hire."}]
    }
    response = await client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "reply" in data
    assert "recommendations" in data
    assert "end_of_conversation" in data
    assert isinstance(data["recommendations"], list)
    # Usually empty if clarifying
    assert len(data["recommendations"]) == 0

@pytest.mark.asyncio
async def test_chat_refusal(client):
    """Test that off-topic queries are refused with schema-compliant response."""
    payload = {
        "messages": [{"role": "user", "content": "Who is the president of France?"}]
    }
    response = await client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "recommendations" in data
    assert len(data["recommendations"]) == 0
    assert "I can only" in data["reply"]

@pytest.mark.asyncio
async def test_chat_missing_messages(client):
    """Test handling of empty message list."""
    payload = {"messages": []}
    response = await client.post("/chat", json=payload)
    # Should be 422 Unprocessable Entity due to Pydantic validation
    assert response.status_code == 422
