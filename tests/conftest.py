"""
tests/conftest.py — Shared pytest fixtures for Auto-CRO test suite.

Agent Tests owns this file.
"""
from __future__ import annotations

import pytest
import torch

from contracts import FEATURE_DIM, N_ARMS
from ml_core.bandits.thompson import ThompsonBandit


@pytest.fixture
def bandit() -> ThompsonBandit:
    """Fresh ThompsonBandit instance with project-wide dimensions."""
    return ThompsonBandit(feature_dim=FEATURE_DIM, n_arms=N_ARMS)


@pytest.fixture
def context() -> torch.Tensor:
    """Random context tensor of shape [1, FEATURE_DIM]."""
    return torch.rand(1, FEATURE_DIM, dtype=torch.float32)


@pytest.fixture
def zero_context() -> torch.Tensor:
    """All-zeros context tensor — baseline for deterministic tests."""
    return torch.zeros(1, FEATURE_DIM, dtype=torch.float32)

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from backend.app import create_app
from ml_core.vlm.extractor import UIFeatures

@pytest.fixture
def mock_extractor():
    """Patches VLMFeatureExtractor.extract() returning a Dummy UIFeatures."""
    dummy_features = UIFeatures(
        is_button_visible=True,
        button_color_hex="#FF0000",
        text_sentiment=0.8,
        visual_clutter=0.2
    )
    with patch("ml_core.vlm.extractor.VLMFeatureExtractor.extract", new_callable=AsyncMock, return_value=dummy_features) as mock:
        yield mock

@pytest.fixture
def client_with_mock(mock_extractor) -> TestClient:
    """TestClient with mocked extractor."""
    app = create_app()
    return TestClient(app)
