"""FastAPI application entry point, lifecycle management, and health probes."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import router as api_router
from .astra_store import AstraStore, build_astra_store
from .config import Settings, load_settings
from .controller import JobController
from .cos_client import build_cos_client
from .jobs import JobStore

logger = logging.getLogger("astra_ingest.main")


def create_app(
    settings: Optional[Settings] = None,
    astra_store: Optional[AstraStore] = None,
    cos_client: Optional[Any] = None,
) -> FastAPI:
    """Application factory for astra-ingest web app."""
    loaded_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        store = astra_store or build_astra_store(loaded_settings)
        cos = cos_client or build_cos_client(loaded_settings)
        job_store = JobStore(store, loaded_settings)
        controller = JobController(loaded_settings, job_store, store, cos)

        app.state.settings = loaded_settings
        app.state.astra_store = store
        app.state.cos_client = cos
        app.state.job_store = job_store
        app.state.controller = controller

        # Recover jobs on startup
        try:
            recovered = controller.recover_jobs()
            if recovered:
                logger.info("Recovered and resumed %d unowned active jobs: %s", len(recovered), recovered)
        except Exception as exc:
            logger.error("Error during startup job recovery: %s", exc)

        yield

        # Shutdown
        logger.info("Gracefully shutting down worker threads...")
        controller.shutdown(timeout=25.0)

    app = FastAPI(
        title="Astra Ingest Web App",
        description="High-throughput document ingestion from IBM Cloud Object Storage into DataStax Astra DB",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Health probes
    @app.get("/healthz", tags=["health"])
    def liveness() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    def readiness(response: Response) -> Dict[str, Any]:
        store = getattr(app.state, "astra_store", None)
        cos = getattr(app.state, "cos_client", None)
        if not store or not cos:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not ready", "error": "Clients not initialized"}
        return {"status": "ready"}

    # Mount API router
    app.include_router(api_router, prefix="/api")

    # Static assets and UI dashboard
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h1>Astra DB Ingestion Service</h1><p>Active.</p>")

    return app


# Default app instance for uvicorn
try:
    app = create_app()
except Exception:
    # If environment variables are not set during import time (e.g. testing)
    app = FastAPI()
