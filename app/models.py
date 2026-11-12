"""Domain Data Models & Serializers for OMNICAST (app/models.py)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.db import db_conn, row_to_dict


class SegmentUpdateRequest(BaseModel):
    tgt_text: str = Field(..., min_length=1, description="Edited target language translation text")


class RegenerateRequest(BaseModel):
    tgt_text: str | None = None
    seed: int | None = None


def list_projects(storage_root=None) -> list[dict[str, Any]]:
    with db_conn(storage_root) as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY created_utc DESC").fetchall()
    return [row_to_dict(r) for r in rows if r is not None]


def get_project_detail(project_id: str, storage_root=None) -> dict[str, Any] | None:
    with db_conn(storage_root) as conn:
        proj_row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not proj_row:
            return None
        proj = row_to_dict(proj_row)
        seg_rows = conn.execute(
            "SELECT * FROM segments WHERE project_id = ? ORDER BY idx ASC", (project_id,)
        ).fetchall()
        rep_rows = conn.execute(
            "SELECT * FROM quality_reports WHERE project_id = ? ORDER BY segment_idx ASC, version ASC, id ASC",
            (project_id,),
        ).fetchall()
        act_rows = conn.execute(
            "SELECT * FROM repair_actions WHERE project_id = ? ORDER BY segment_idx ASC, attempt ASC, id ASC",
            (project_id,),
        ).fetchall()

    reports_by_seg: dict[int, list[dict]] = {}
    for r in rep_rows:
        rd = row_to_dict(r)
        reports_by_seg.setdefault(rd["segment_idx"], []).append(rd)

    actions_by_seg: dict[int, list[dict]] = {}
    for a in act_rows:
        ad = row_to_dict(a)
        actions_by_seg.setdefault(ad["segment_idx"], []).append(ad)

    segments: list[dict[str, Any]] = []
    for s in seg_rows:
        sd = row_to_dict(s)
        idx = sd["idx"]
        seg_reps = reports_by_seg.get(idx, [])
        active_ver = sd["active_version"]
        active_rep = next(
            (r for r in reversed(seg_reps) if r["version"] == active_ver),
            seg_reps[-1] if seg_reps else None,
        )
        sd["active_quality"] = active_rep
        sd["quality_history"] = seg_reps
        sd["repair_actions"] = actions_by_seg.get(idx, [])
        segments.append(sd)

    proj["segments"] = segments
    proj["repair_actions"] = [row_to_dict(a) for a in act_rows]
    proj["summary"] = {
        "total_segments": len(segments),
        "accepted_segments": sum(1 for s in segments if s["status"] == "ACCEPTED"),
        "repaired_segments": sum(1 for s in segments if s["status"] == "REPAIRED"),
        "human_review_segments": sum(1 for s in segments if s["status"] == "HUMAN_REVIEW"),
        "total_repair_attempts": sum(1 for a in proj["repair_actions"] if a["action"] != "HUMAN_REVIEW"),
        "accepted_repairs": sum(1 for a in proj["repair_actions"] if a["accepted"]),
    }
    return proj
