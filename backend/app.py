"""
backend/app.py — FastAPI application factory.

Agent Backend owns this file.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from dotenv import load_dotenv

# Load environment variables from .env file FIRST before any other project imports
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.api import routes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize resources on startup, clean up on shutdown."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Auto-CRO backend starting. Warming up cache...")
    from backend.api.deps import get_bandit
    get_bandit()  # прогрев кеша
    
    from ml_core.mlops import init_mlflow
    init_mlflow(run_name="production-run")
    
    logger.info("Auto-CRO backend started.")
    yield
    
    from ml_core.mlops import finish
    finish()
    
    logger.info("Auto-CRO backend shutting down.")


def create_app() -> FastAPI:
    """
    Application factory.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title="Auto-CRO API",
        description="Contextual Thompson Sampling for UI optimization",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # TODO (Agent Backend): сузить в prod
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes.router)

    # Serve SDK static files
    frontend_js = Path("frontend/js").resolve()
    frontend_templates = Path("frontend/templates").resolve()
    
    app.mount("/static", StaticFiles(directory=str(frontend_js)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(frontend_templates / "index.html"))

    return app


# Entry point for `uvicorn backend.app:app`
app = create_app()
