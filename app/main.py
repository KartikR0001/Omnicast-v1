"""FastAPI web application entry point for OMNICAST Phase-1."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ai.registry import ModelRegistry
from app.api.export import router as export_router
from app.api.projects import router as projects_router
from app.api.segments import router as segments_router
from app.db import init_db
from app.models import get_project_detail, list_projects
from app.worker import JobWorker

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"


def create_app(
    storage_dir: Path | str = Path("storage"),
    registry: ModelRegistry | None = None,
    sync_jobs: bool = False,
) -> FastAPI:
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    init_db(storage_path)

    reg = registry or ModelRegistry()
    worker = None if sync_jobs else JobWorker(registry=reg, storage_root=storage_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from pipeline.ingest import _tool_path
        for tool in ("ffmpeg", "ffprobe"):
            try:
                _tool_path(tool)
            except RuntimeError:
                print(f"[OMNICAST] WARNING: '{tool}' not found. Uploads will fail until ffmpeg is installed "
                      "(brew install ffmpeg) and the app is restarted.", flush=True)
        try:
            yield
        finally:
            if worker is not None:
                worker.close()
            elif registry is None:
                reg.close()

    app = FastAPI(
        title="OMNICAST Phase-1 Studio",
        description="End-to-End AI Studio for Long-Form Audio Production and Multilingual Distribution",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.storage_dir = storage_path
    app.state.registry = reg
    app.state.worker = worker
    app.state.sync_jobs = sync_jobs

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.include_router(projects_router)
    app.include_router(segments_router)
    app.include_router(export_router)

    @app.get("/", response_class=HTMLResponse)
    def ui_index(request: Request) -> Any:
        projects = list_projects(app.state.storage_dir)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"projects": projects},
        )

    @app.get("/projects/{project_id}", response_class=HTMLResponse)
    def ui_project_review(project_id: str, request: Request) -> Any:
        project = get_project_detail(project_id, app.state.storage_dir)
        if not project:
            return HTMLResponse(
                content=f"<h1>404 — Project {project_id} not found</h1>",
                status_code=404,
            )
        return templates.TemplateResponse(
            request,
            "review.html",
            {"project": project},
        )

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ok"})

    return app


app = create_app()
