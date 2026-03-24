"""
tests/test_api.py — Integration tests for FastAPI endpoints.

Tests: /health, /decide (501 until wired), /feedback (422 validation).
Uses FastAPI TestClient (sync wrapper over ASGI).
Agent Tests owns this file.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestDecideEndpoint:
    def test_decide_without_image_returns_422(self, client: TestClient) -> None:
        """POST /decide without image file should return 422 validation error."""
        response = client.post("/decide")
        assert response.status_code == 422

    def test_decide_with_image_returns_200(self, client_with_mock: TestClient) -> None:
        """POST /decide with image returns 200 and valid schema."""
        fake_image = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        response = client_with_mock.post(
            "/decide",
            files={"image": ("test.jpg", fake_image, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "arm_index" in data
        assert "confidence" in data
        assert "session_id" in data


class TestFeedbackEndpoint:
    def test_feedback_valid_payload_accepted(self, client: TestClient) -> None:
        """POST /feedback with valid payload should be accepted."""
        from backend.api.routes import _CONTEXT_STORE
        import torch
        
        _CONTEXT_STORE["test_sess"] = (0, torch.zeros(1, 4))
        
        response = client.post(
            "/feedback",
            json={"session_id": "test_sess", "reward": 1.0},
        )
        assert response.status_code == 200

    def test_feedback_invalid_reward_returns_422(self, client: TestClient) -> None:
        """POST /feedback with reward > 1.0 should return 422."""
        response = client.post(
            "/feedback",
            json={"session_id": "test_sess", "reward": 2.5},
        )
        assert response.status_code == 422

    def test_feedback_missing_session_id_returns_422(self, client: TestClient) -> None:
        """POST /feedback without session_id should return 422."""
        response = client.post(
            "/feedback",
            json={"reward": 1.0},
        )
        assert response.status_code == 422
