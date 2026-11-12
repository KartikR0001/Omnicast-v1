"""SQLite Persistence Layer for OMNICAST Studio (app/db.py).

Stores Projects, Segments, QualityReports (D2), and RepairActions (D4).
Uses WAL mode so background worker writes never block FastAPI status polling (R10).
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE = Path(os.environ.get("OMNICAST_STORAGE", ROOT / "storage"))

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    work_wav_path TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    tts_provider TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    error_reason TEXT,
    duration_s REAL,
    sample_rate INTEGER,
    channels INTEGER,
    size_bytes INTEGER,
    sha256 TEXT,
    out_wav_path TEXT,
    out_srt_path TEXT,
    out_duration_s REAL,
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    out_start_s REAL,
    out_end_s REAL,
    src_text TEXT NOT NULL,
    tgt_text TEXT NOT NULL,
    src_wav_path TEXT NOT NULL,
    active_version INTEGER NOT NULL DEFAULT 1,
    active_audio_path TEXT NOT NULL,
    status TEXT NOT NULL,
    asr_seconds REAL,
    mt_seconds REAL,
    tts_seconds REAL,
    UNIQUE(project_id, idx)
);

CREATE TABLE IF NOT EXISTS quality_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_idx INTEGER NOT NULL,
    version INTEGER NOT NULL,
    audio_path TEXT NOT NULL,
    src_duration_s REAL,
    gen_duration_s REAL,
    cer REAL,
    duration_ratio REAL,
    clipping_fraction REAL,
    silence_ratio REAL,
    empty_audio INTEGER NOT NULL,
    bt_chrf REAL,
    eval_transcript TEXT,
    flags_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repair_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_idx INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    verification_note TEXT NOT NULL,
    candidate_version INTEGER NOT NULL,
    active_version_after INTEGER NOT NULL,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def get_storage_root(custom_root: Path | str | None = None) -> Path:
    root = Path(custom_root) if custom_root else Path(os.environ.get("OMNICAST_STORAGE", DEFAULT_STORAGE))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_db_path(storage_root: Path | str | None = None) -> Path:
    return get_storage_root(storage_root) / "omnicast.db"


def init_db(storage_root: Path | str | None = None) -> Path:
    db_path = get_db_path(storage_root)
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        conn.executescript(SCHEMA_SQL)
    return db_path


@contextmanager
def db_conn(storage_root: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    db_path = init_db(storage_root)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for k in ("flags_json", "before_json", "after_json"):
        if k in d and isinstance(d[k], str):
            try:
                d[k.replace("_json", "")] = json.loads(d[k])
            except json.JSONDecodeError:
                d[k.replace("_json", "")] = d[k]
    if "empty_audio" in d:
        d["empty_audio"] = bool(d["empty_audio"])
    if "accepted" in d:
        d["accepted"] = bool(d["accepted"])
    return d

