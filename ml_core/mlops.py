"""
ml_core/mlops.py — MLflow experiment tracking for Auto-CRO.

Provides structured logging of each optimization step including
chosen arm, reward signal, and uncertainty estimate.
"""
from __future__ import annotations

import logging
from typing import Any
import os

from contracts import StepLog

logger = logging.getLogger(__name__)

_mlflow_enabled: bool = False


def init_mlflow(
    experiment_name: str = "auto-cro-experiment",
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Initialize an MLflow run for experiment tracking.
    """
    global _mlflow_enabled
    if config is None:
        config = {}
        
    from contracts import FEATURE_DIM, N_ARMS
    config.setdefault("FEATURE_DIM", FEATURE_DIM)
    config.setdefault("N_ARMS", N_ARMS)
    config.setdefault("model_name", "google/gemini-2.0-flash-001")
    
    try:
        import mlflow  # noqa: PLC0415
        
        # Set absolute path for local MLflow tracking or remote URI
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        active_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(config)
        
        _mlflow_enabled = True
        logger.info("MLflow initialized: experiment=%s, run_id=%s", experiment_name, active_run.info.run_id)
    except ImportError:
        logger.warning("mlflow not installed — logging disabled")
    except Exception as exc:
        logger.warning("MLflow init failed: %s — continuing without tracking", exc)


def log_step(step_log: StepLog) -> None:
    """
    Log a single optimization step to MLflow.
    """
    if not _mlflow_enabled:
        logger.debug("MLflow disabled — skipping log_step")
        return

    try:
        import mlflow  # noqa: PLC0415
        
        # Log metrics for this step
        metrics = {
            "chosen_arm": step_log["chosen_arm"],
            "reward": step_log["reward"],
            "uncertainty": step_log["uncertainty"]
        }
        
        # In MLflow we pass the incremental step explicitly
        mlflow.log_metrics(metrics, step=step_log["step"])
        
    except Exception as exc:
        logger.warning("MLflow log step failed: %s", exc)


def finish() -> None:
    """Finalize the MLflow run."""
    if not _mlflow_enabled:
        return
    try:
        import mlflow  # noqa: PLC0415
        mlflow.end_run()
        logger.info("MLflow run finished.")
    except Exception as exc:
        logger.warning("MLflow finish failed: %s", exc)
