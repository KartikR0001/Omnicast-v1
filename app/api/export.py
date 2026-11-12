"""Export endpoints: download reconstructed WAV and aligned SRT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from app.db import db_conn

router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["export"])


@router.get("/wav")
def api_export_wav(project_id: str, request: Request) -> Any:
    storage_dir: Path = request.app.state.storage_dir
    with db_conn(storage_dir) as conn:
        proj = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()

    if not proj:
        return JSONResponse(
            status_code=404,
            content={"error_code": "PROJECT_NOT_FOUND", "message": f"Project {project_id} not found"},
        )

    wav_path_str = proj["out_wav_path"]
    if not wav_path_str or not Path(wav_path_str).exists():
        return JSONResponse(
            status_code=409,
            content={"error_code": "EXPORT_NOT_READY", "message": "Exported WAV is not yet available"},
        )

    return FileResponse(
        Path(wav_path_str),
        media_type="audio/wav",
        filename=f"omnicast_{project_id}_{proj['target_lang']}.wav",
    )


@router.get("/srt")
def api_export_srt(project_id: str, request: Request) -> Any:
    storage_dir: Path = request.app.state.storage_dir
    with db_conn(storage_dir) as conn:
        proj = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()

    if not proj:
        return JSONResponse(
            status_code=404,
            content={"error_code": "PROJECT_NOT_FOUND", "message": f"Project {project_id} not found"},
        )

    srt_path_str = proj["out_srt_path"]
    if not srt_path_str or not Path(srt_path_str).exists():
        return JSONResponse(
            status_code=409,
            content={"error_code": "EXPORT_NOT_READY", "message": "Exported SRT is not yet available"},
        )

    return FileResponse(
        Path(srt_path_str),
        media_type="application/x-subrip; charset=utf-8",
        filename=f"omnicast_{project_id}_{proj['target_lang']}.srt",
    )
