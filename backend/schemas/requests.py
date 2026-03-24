"""
backend/schemas/requests.py — Pydantic input schemas for API endpoints.

Agent Backend owns this file.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from contracts import ArmIndex, Reward


class FeedbackRequest(BaseModel):
    """
    Request body for POST /feedback.

    Attributes:
        arm_index: Index of the arm that was shown to the user.
        reward: Conversion signal — 1.0 for click/conversion, 0.0 for no action.
    """

    session_id: str = Field(..., description="Unique session ID returned by /decide")
    reward: Reward = Field(..., ge=0.0, le=1.0, description="Conversion signal: 1.0=converted, 0.0=not")

    @field_validator("reward")
    @classmethod
    def reward_must_be_binary(cls, v: float) -> float:
        """Ensure reward is 0.0 or 1.0 for binary feedback or keep continuous."""
        return v
