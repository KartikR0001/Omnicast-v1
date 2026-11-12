"""Segment endpoints: edit translation, regenerate segment, stream segment audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from app.db import db_conn, utc_now
from app.models import RegenerateRequest, SegmentUpdateRequest, get_project_detail
from app.worker import regenerate_single_segment

router = APIRouter(prefix="/api/projects/{project_id}/segments", tags=["segments"])


@router.patch("/{segment_idx}")
def api_update_segment_text(
    project_id: str,
    segment_idx: int,
    payload: SegmentUpdateRequest,
    request: Request,
) -> JSONResponse:
    storage_dir: Path = request.app.state.storage_dir
    with db_conn(storage_dir) as conn:
        seg = conn.execute(
            "SELECT * FROM segments WHERE project_id = ? AND idx = ?",
            (project_id, segment_idx),
        ).fetchone()
        if not seg:
            return JSONResponse(
                status_code=404,
                content={"error_code": "SEGMENT_NOT_FOUND", "message": f"Segment {segment_idx} not found"},
            )
        conn.execute(
            "UPDATE segments SET tgt_text = ? WHERE id = ?",
            (payload.tgt_text.strip(), seg["id"]),
        )
        conn.execute(
            "UPDATE projects SET updated_utc = ? WHERE id = ?",
            (utc_now(), project_id),
        )
    detail = get_project_detail(project_id, storage_dir)
    return JSONResponse(status_code=200, content=detail)


@router.post("/{segment_idx}/regenerate")
def api_regenerate_segment(
    project_id: str,
    segment_idx: int,
    request: Request,
    payload: RegenerateRequest | None = None,
) -> JSONResponse:
    storage_dir: Path = request.app.state.storage_dir
    worker = getattr(request.app.state, "worker", None)
    registry = worker.registry if worker is not None else request.app.state.registry

    try:
        seg_detail = regenerate_single_segment(
            project_id=project_id,
            segment_idx=segment_idx,
            new_tgt_text=payload.tgt_text if payload else None,
            seed=payload.seed if payload else None,
            registry=registry,
            storage_root=storage_dir,
        )
    except KeyError as exc:
        return JSONResponse(
            status_code=404,
            content={"error_code": "NOT_FOUND", "message": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error_code": "REGENERATE_ERROR", "message": f"{type(exc).__name__}: {exc}"},
        )
    detail = get_project_detail(project_id, storage_dir)
    return JSONResponse(status_code=200, content={"segment": seg_detail, "project": detail})


@router.get("/{segment_idx}/audio/{kind}")
def api_get_segment_audio(
    project_id: str,
    segment_idx: int,
    kind: str,
    request: Request,
) -> Any:
    if kind not in ("source", "target"):
        return JSONResponse(
            status_code=400,
            content={"error_code": "INVALID_AUDIO_KIND", "message": "kind must be 'source' or 'target'"},
        )
    storage_dir: Path = request.app.state.storage_dir
    with db_conn(storage_dir) as conn:
        seg = conn.execute(
            "SELECT * FROM segments WHERE project_id = ? AND idx = ?",
            (project_id, segment_idx),
        ).fetchone()

    if not seg:
        return JSONResponse(
            status_code=404,
            content={"error_code": "SEGMENT_NOT_FOUND", "message": f"Segment {segment_idx} not found"},
        )

    wav_path_str = seg["src_wav_path"] if kind == "source" else seg["active_audio_path"]
    if not wav_path_str or not Path(wav_path_str).exists():
        return JSONResponse(
            status_code=404,
            content={"error_code": "AUDIO_NOT_FOUND", "message": f"{kind} audio not available for segment {segment_idx}"},
        )
    return FileResponse(
        Path(wav_path_str),
        media_type="audio/wav",
        filename=f"{project_id}_seg_{segment_idx:03d}_{kind}.wav",
    )

