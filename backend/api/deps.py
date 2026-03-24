"""
backend/api/deps.py — Dependency injection for FastAPI.

Provides a singleton ThompsonBandit instance shared across all requests.
Agent Backend owns this file.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from contracts import FEATURE_DIM, N_ARMS
from ml_core.bandits.thompson import ThompsonBandit

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_bandit() -> ThompsonBandit:
    """
    Returns the global ThompsonBandit singleton.

    Uses lru_cache to ensure only one instance is created across all requests.
    Cache is invalidated via `get_bandit.cache_clear()` if needed (e.g., in tests).

    Returns:
        ThompsonBandit: Initialized bandit with project-wide FEATURE_DIM and N_ARMS.
    """
    logger.info("Initializing ThompsonBandit singleton (feature_dim=%d, n_arms=%d)", FEATURE_DIM, N_ARMS)
    bandit = ThompsonBandit(feature_dim=FEATURE_DIM, n_arms=N_ARMS)
    
    # Try to load state from Supabase
    try:
        from ml_core.storage import load_state, get_supabase_client
        supabase_client = get_supabase_client()
        loaded_any = False
        for i in range(N_ARMS):
            state = load_state(supabase_client, i)
            if state is not None:
                A, b = state
                bandit.A[i] = A
                bandit.b[i] = b
                loaded_any = True
        
        if loaded_any:
            logger.info("Successfully loaded bandit state from Supabase.")
    except Exception as e:
        logger.warning("Could not load bandit state from Supabase (this is expected if MLOps is pending or credentials missing): %s", e)

    return bandit

@lru_cache(maxsize=1)
def get_supabase() -> "Any":
    """Dependency injecting Supabase client."""
    try:
        from ml_core.storage import get_supabase_client
        return get_supabase_client()
    except Exception as e:
        logger.warning("Supabase client init failed: %s", e)
        return None
