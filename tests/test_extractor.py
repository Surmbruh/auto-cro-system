"""
tests/test_extractor.py — Unit tests for VLMFeatureExtractor.

Tests: successful extraction, HTTP error handling, Pydantic validation failure.
All tests use mocked httpx to avoid real API calls.
Agent Tests owns this file.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ml_core.vlm.extractor import VLMFeatureExtractor, UIFeatures


VALID_VLM_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": json.dumps({
                    "is_button_visible": True,
                    "button_color_hex": "#FF5733",
                    "text_sentiment": 0.5,
                    "visual_clutter": 0.3,
                })
            }
        }
    ]
}

FAKE_IMAGE_BYTES: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header


@pytest.fixture
def extractor() -> VLMFeatureExtractor:
    return VLMFeatureExtractor(api_key="test-key-12345")


@pytest.mark.asyncio
async def test_extract_success(extractor: VLMFeatureExtractor) -> None:
    """extract() should return a valid UIFeatures on successful API response."""
    mock_response = MagicMock()
    mock_response.json.return_value = VALID_VLM_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await extractor.extract(FAKE_IMAGE_BYTES)

    assert isinstance(result, UIFeatures)
    assert result.is_button_visible is True
    assert result.button_color_hex == "#FF5733"
    assert result.text_sentiment == pytest.approx(0.5)
    assert result.visual_clutter == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_extract_http_error_raises_runtime_error(extractor: VLMFeatureExtractor) -> None:
    """extract() should raise RuntimeError on HTTP failure."""
    import httpx

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.HTTPError("timeout")):
        with pytest.raises(RuntimeError, match="Ошибка вызова OpenRouter API"):
            await extractor.extract(FAKE_IMAGE_BYTES)


@pytest.mark.asyncio
async def test_extract_invalid_json_raises_value_error(extractor: VLMFeatureExtractor) -> None:
    """extract() should raise ValueError if VLM returns invalid JSON."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "not a json string {"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        with pytest.raises(ValueError, match="Ошибка парсинга"):
            await extractor.extract(FAKE_IMAGE_BYTES)


@pytest.mark.asyncio
async def test_extract_missing_field_raises_value_error(extractor: VLMFeatureExtractor) -> None:
    """extract() should raise ValueError if required UIFeatures field is missing."""
    mock_response = MagicMock()
    # Missing 'visual_clutter' field
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"is_button_visible": true, "button_color_hex": "#FF0000", "text_sentiment": 0.1}'}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        with pytest.raises(ValueError):
            await extractor.extract(FAKE_IMAGE_BYTES)

@pytest.mark.asyncio
async def test_extract_passes_base64_image(extractor: VLMFeatureExtractor) -> None:
    """extract() should pass base64 encoded image in the payload."""
    mock_response = MagicMock()
    mock_response.json.return_value = VALID_VLM_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        await extractor.extract(FAKE_IMAGE_BYTES)
        
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        
        import base64
        expected_b64 = base64.b64encode(FAKE_IMAGE_BYTES).decode('utf-8')
        assert image_url == f"data:image/jpeg;base64,{expected_b64}"

def test_extractor_default_model() -> None:
    """Check default model is 'google/gemini-2.0-flash-001'"""
    e = VLMFeatureExtractor(api_key="123")
    assert e.model == "google/gemini-2.0-flash-001"
