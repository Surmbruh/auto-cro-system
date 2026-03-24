"""
backend/api/routes.py — HTTP route handlers.

Endpoints:
    POST /decide   — accept UI screenshot, return best arm index
    POST /feedback — accept reward signal, update bandit
    GET  /health   — liveness probe

Agent Backend owns this file.
"""
from __future__ import annotations

import logging
import os
import uuid
from cachetools import TTLCache

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from backend.api.deps import get_bandit
from backend.schemas.requests import FeedbackRequest
from backend.schemas.responses import DecideResponse, HealthResponse
from contracts import ArmIndex, ContextTensor
from ml_core.bandits.thompson import ThompsonBandit
from main_pipeline import run_optimization_step

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory context store with automatic Garbage Collection
# Stores tuple: (arm_index: int, context: ContextTensor)
# maxsize: max 10,000 active sessions, ttl: expire after 1 hour (3600 seconds)
_CONTEXT_STORE = TTLCache(maxsize=10000, ttl=3600)


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok")


@router.post("/decide", response_model=DecideResponse, tags=["optimization"])
async def decide(
    image: UploadFile = File(..., description="UI screenshot (JPEG/PNG)"),
    bandit: ThompsonBandit = Depends(get_bandit),
) -> DecideResponse:
    """
    Accept a UI screenshot and return the optimal arm index.

    Args:
        image: Screenshot of the current UI state.
        bandit: Injected ThompsonBandit singleton.

    Returns:
        DecideResponse with chosen arm_index and confidence score.

    Raises:
        HTTPException 400: If image cannot be read.
        HTTPException 500: If VLM extraction or bandit sampling fails.
    """
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image: {e}",
        )
        
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is missing from environment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: missing API key.",
        )

    try:
        chosen_arm, context = await run_optimization_step(api_key, image_bytes)
    except Exception as e:
        logger.exception("run_optimization_step failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization pipeline error: {e}",
        )

    # Generate unique session ID to track this specific user context with TTL
    session_id = str(uuid.uuid4())
    _CONTEXT_STORE[session_id] = (chosen_arm, context)

    return DecideResponse(session_id=session_id, arm_index=chosen_arm, confidence=0.0)


from backend.api.deps import get_bandit, get_supabase

# ... in feedback endpoint ...
@router.post("/feedback", tags=["optimization"])
async def feedback(
    payload: FeedbackRequest,
    bandit: ThompsonBandit = Depends(get_bandit),
    supabase_client: "Any" = Depends(get_supabase)
) -> dict[str, str]:
    """
    Accept a reward signal and update the bandit posterior.

    Args:
        payload: FeedbackRequest containing session_id and reward.
        bandit: Injected ThompsonBandit singleton.
        supabase_client: Supabase client to sync db.

    Returns:
        Confirmation dict {"status": "ok"}.

    Raises:
        HTTPException 404: If context is not found for the given session_id.
        HTTPException 500: If bandit update fails.
    """
    session_data = _CONTEXT_STORE.get(payload.session_id)
    if session_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context not found or expired for session_id {payload.session_id}. Cannot process feedback.",
        )
    arm_index, context = session_data

    try:
        bandit.update(arm_idx=arm_index, context=context, reward=payload.reward)
        if supabase_client:
            bandit.sync_with_db(supabase_client)
            
        # Logging to MLflow
        import time
        import torch
        from ml_core.mlops import log_step
        
        # Calculate uncertainty for logging
        A_inv = torch.linalg.inv(bandit.A[arm_index])
        # context is [1, d] tensor
        uncertainty = torch.sqrt(torch.matmul(torch.matmul(context, A_inv), context.T)).item()
        
        log_step({
            "step": int(time.time()), 
            "chosen_arm": arm_index,
            "reward": payload.reward,
            "uncertainty": uncertainty,
            "model_name": "google/gemini-2.0-flash-001"
        })
        
    except Exception as e:
        logger.exception("Bandit update failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bandit update failed: {e}",
        )

    # Clean up the context once used
    _CONTEXT_STORE.pop(payload.session_id, None)

    return {"status": "ok"}
