"""
ml_core/storage.py — Supabase serialization of ThompsonBandit state.

Handles save/load of PyTorch tensors (A matrix, b vector) using
base64 encoding for storage in Supabase bandit_states table.

Agent MLOps owns this file.
Table schema: bandit_states(arm_index INT, A_bytes TEXT, b_bytes TEXT, updated_at TIMESTAMPTZ)
"""
from __future__ import annotations

import base64
import io
import logging
import os
from datetime import datetime, timezone

import torch

from contracts import ArmIndex, BanditState

logger = logging.getLogger(__name__)


def _tensor_to_b64(tensor: torch.Tensor) -> str:
    """
    Serialize a PyTorch tensor to a base64-encoded string.

    Args:
        tensor: Any PyTorch tensor.

    Returns:
        Base64-encoded string representation of the tensor.
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _b64_to_tensor(b64_str: str) -> torch.Tensor:
    """
    Deserialize a base64-encoded string back to a PyTorch tensor.

    Args:
        b64_str: Base64-encoded tensor string from storage.

    Returns:
        Deserialized PyTorch tensor.
    """
    raw = base64.b64decode(b64_str)
    buffer = io.BytesIO(raw)
    import typing
    return typing.cast(torch.Tensor, torch.load(buffer, weights_only=True))


from typing import Any

def get_supabase_client() -> Any:
    from supabase import create_client
    url: str = os.environ.get("SUPABASE_URL", "")
    key: str = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    return create_client(url, key)


def save_state(supabase_client: Any, A: torch.Tensor, b: torch.Tensor, arm_idx: ArmIndex) -> None:
    """
    Serialize and upsert bandit matrices for a given arm to Supabase.

    Args:
        supabase_client: Initialized Supabase client instance.
        A: Precision matrix A[arm_idx] of shape [d, d].
        b: Reward vector b[arm_idx] of shape [d, 1].
        arm_idx: Index of the arm whose state to save.
    """
    state: BanditState = {
        "arm_index": arm_idx,
        "a_bytes": _tensor_to_b64(A),
        "b_bytes": _tensor_to_b64(b),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.debug("Saving state for arm %d to Supabase", arm_idx)
    try:
        supabase_client.table("bandit_states").upsert(state, on_conflict="arm_index").execute()
    except Exception as e:
        logger.warning(f"Failed to save state to Supabase: {e}")


def load_state(supabase_client: Any, arm_idx: ArmIndex) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load and deserialize bandit matrices for a given arm from Supabase.

    Args:
        supabase_client: Initialized Supabase client instance.
        arm_idx: Index of the arm to load.

    Returns:
        Tuple (A, b) of tensors if found, None otherwise.
    """
    logger.debug("Loading state for arm %d from Supabase", arm_idx)
    try:
        response = supabase_client.table("bandit_states").select("*").eq("arm_index", arm_idx).execute()
        if not hasattr(response, 'data') or not response.data:
            return None
        row = response.data[0]
        return _b64_to_tensor(row["a_bytes"]), _b64_to_tensor(row["b_bytes"])
    except Exception as e:
        logger.warning(f"Failed to load state from Supabase: {e}")
        return None
