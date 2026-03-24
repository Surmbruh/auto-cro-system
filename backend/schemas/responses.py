"""
backend/schemas/responses.py — Pydantic output schemas for API endpoints.

Agent Backend owns this file.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from contracts import ArmIndex


class DecideResponse(BaseModel):
    """
    Response body for POST /decide.

    Attributes:
        arm_index: Index of the UI variant chosen by the bandit.
        confidence: Sampled expected reward — higher means more certainty.
    """

    session_id: str = Field(..., description="Unique session ID for providing feedback")
    arm_index: ArmIndex = Field(..., description="Chosen UI variant index (0-based)")
    confidence: float = Field(..., description="Sampled expected reward from Thompson Sampling")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(default="ok")
