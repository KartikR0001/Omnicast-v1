"""Project endpoints: upload, list, status, and detail."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from ai.tts.chatterbox_client import PY as CHATTERBOX_PY
from app.models import get_project_detail, list_projects
from app.worker import create_project_from_file, run_pipeline_job
from pipeline.ingest import MAX_BYTES, MediaValidationError

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("")
def api_list_projects(request: Request) -> list[dict[str, Any]]:
    storage_dir: Path = request.app.state.storage_dir
    return list_projects(storage_dir)


@router.get("/{project_id}")
def api_get_project(project_id: str, request: Request) -> JSONResponse:
    storage_dir: Path = request.app.state.storage_dir
    detail = get_project_detail(project_id, storage_dir)
    if not detail:
        return JSONResponse(
            status_code=404,
            content={"error_code": "PROJECT_NOT_FOUND", "message": f"Project {project_id} not found"},
        )
    return JSONResponse(status_code=200, content=detail)


@router.post("")
async def api_create_project(
    request: Request,
    file: UploadFile = File(...),
    target_lang: str = Form("hi"),
    tts_provider: str = Form("mms_tts"),
    name: str | None = Form(None),
) -> JSONResponse:
    if target_lang not in ("hi", "fr", "de", "es"):
        return JSONResponse(
            status_code=400,
            content={
                "error_code": "INVALID_TARGET_LANG",
                "message": f"Unsupported target_lang '{target_lang}'. Must be one of: hi, fr, de, es.",
            },
        )
    if tts_provider not in ("mms_tts", "chatterbox"):
        return JSONResponse(
            status_code=400,
            content={
                "error_code": "INVALID_TTS_PROVIDER",
                "message": f"Unsupported tts_provider '{tts_provider}'. Must be 'mms_tts' or 'chatterbox'.",
            },
        )

    if tts_provider == "chatterbox" and not CHATTERBOX_PY.exists():
        return JSONResponse(
            status_code=400,
            content={
                "error_code": "CHATTERBOX_NOT_INSTALLED",
                "message": ("Chatterbox needs its own Python environment at .tools/envs/chatterbox, which is not "
                            "installed on this computer. Choose MMS-TTS instead."),
            },
        )

    storage_dir: Path = request.app.state.storage_dir
    storage_dir.mkdir(parents=True, exist_ok=True)

    original_filename = file.filename or "upload.bin"
    suffix = Path(original_filename).suffix
    tmp_dir = storage_dir / "tmp_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / original_filename

    # Write to a unique temporary file preserving the original filename extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as tmp:
        tmp_path = Path(tmp.name)
        total_written = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_written += len(chunk)
            tmp.write(chunk)
            if total_written > MAX_BYTES + 1024 * 1024:
                break

    try:
        proj = create_project_from_file(
            src_path=tmp_path,
            target_lang=target_lang,
            name=name or Path(original_filename).stem,
            tts_provider=tts_provider,
            storage_root=storage_dir,
        )
        # Update source_filename to reflect the user's uploaded filename
        from app.db import db_conn

        with db_conn(storage_dir) as conn:
            conn.execute(
                "UPDATE projects SET source_filename = ? WHERE id = ?",
                (original_filename, proj["id"]),
            )
        proj["source_filename"] = original_filename
    except MediaValidationError as exc:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "status": "REJECTED",
                "error_code": exc.code,
                "message": exc.reason,
            },
        )
    except RuntimeError as exc:
        # e.g. ffmpeg / ffprobe not installed: report it instead of a bare "Internal Server Error"
        missing_tool = "not found" in str(exc) and ("ffprobe" in str(exc) or "ffmpeg" in str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "FFMPEG_NOT_FOUND" if missing_tool else "SERVER_ERROR",
                "message": ("ffmpeg is not installed or not on PATH. Install it (brew install ffmpeg), open a new "
                            "Terminal and restart the app.") if missing_tool else str(exc),
            },
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    project_id = proj["id"]
    worker = getattr(request.app.state, "worker", None)
    sync_jobs = getattr(request.app.state, "sync_jobs", False)

    if sync_jobs:
        reg = worker.registry if worker is not None else request.app.state.registry
        run_pipeline_job(project_id, registry=reg, storage_root=storage_dir)
    elif worker is not None:
        worker.submit(project_id)

    return JSONResponse(
        status_code=202,
        content={
            "project_id": project_id,
            "id": project_id,
            "status": "QUEUED",
            "stage": "INGESTED",
            "target_lang": target_lang,
            "tts_provider": tts_provider,
            "duration_s": proj["duration_s"],
        },
    )
