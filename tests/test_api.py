"""Tests for the FastAPI web endpoints.

Uses FastAPI's TestClient to send HTTP requests without starting a real server.
These tests verify:
    - GET /health returns 200 and {"status": "ok"}.
    - POST /query returns 200 with an SSE stream for valid input.
    - POST /query returns 400 when the question is empty.
    - POST /query streams a rejected response for off-topic queries.
    - POST /build returns 200 after rebuilding the store.
    - POST /build returns 500 on build failure.

The LangGraph pipeline, vector-store builder, and embedder are all mocked
so tests run instantly without real models or databases.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api import app
import src.api as api_module  # needed to reset the in-memory semantic cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_semantic_cache():
    """Reset the semantic cache before every test to avoid cross-test contamination."""
    api_module._semantic_cache.clear()
    yield
    api_module._semantic_cache.clear()


# ──────────────────────────── Health check ───────────────────────────────────

def test_check_endpoint():
    """GET /health should always return 200 with {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ──────────────────────────── /query endpoint ────────────────────────────────

@patch('src.api.graph.astream_events')
@patch('src.api.OllamaEmbedder.embed')
def test_valid_query(mock_embed, mock_astream):
    """A valid academic question should return 200 with an SSE stream containing
    the mocked response and citations."""
    # Return a fixed embedding so the cache logic runs without a real model
    mock_embed.return_value = [0.1] * 768

    # Simulate LangGraph emitting a final on_chain_end event
    async def fake_events():
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {
                "output": {
                    "response": "ALS is caused by mutations.",
                    "citations": [{"id": "doc1"}],
                    "rejected": False
                }
            }
        }

    mock_astream.return_value = fake_events()

    response = client.post("/query", json={"question": "What causes ALS?"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    content = response.content.decode()
    assert "data: " in content
    assert "ALS is caused by mutations." in content
    assert "doc1" in content


def test_empty_query():
    """An empty question string should be rejected immediately with HTTP 400."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400


@patch('src.api.graph.astream_events')
@patch('src.api.OllamaEmbedder.embed')
def test_rejected_query(mock_embed, mock_astream):
    """An off-topic question should stream back a rejection message (not cached)."""
    # Return a zero vector so cosine similarity stays at 0 — avoids any cache hit
    mock_embed.return_value = [0.0] * 768

    async def fake_events():
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {
                "output": {
                    "response": "Please ask an academic question.",
                    "citations": [],
                    "rejected": True
                }
            }
        }

    mock_astream.return_value = fake_events()

    response = client.post("/query", json={"question": "Best pizza in Rome?"})

    assert response.status_code == 200
    content = response.content.decode()
    assert "Please ask an academic question." in content


# ──────────────────────────── /build endpoint ────────────────────────────────

@patch('src.api.build_vector_store')
def test_build_success(mock_build):
    """POST /build should call build_vector_store() and return a success message."""
    response = client.post("/build", json={})
    assert response.status_code == 200
    assert "Vector store rebuilt successfully" in response.json()["status"]
    mock_build.assert_called_once()


@patch('src.api.build_vector_store')
def test_build_failure(mock_build):
    """POST /build should return HTTP 500 with the error detail on failure."""
    mock_build.side_effect = Exception("Build failed")
    response = client.post("/build", json={})
    assert response.status_code == 500
    assert "Build failed" in response.json()["detail"]