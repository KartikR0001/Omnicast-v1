"""Command-line interface for OMNICAST Phase-1 pipeline and repair engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai.registry import ModelRegistry
from app.db import init_db
from app.models import get_project_detail
from app.worker import (
    create_project_from_file,
    regenerate_single_segment,
    run_pipeline_job,
)
from pipeline.ingest import MediaValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnicast",
        description="OMNICAST Phase-1 Media Pipeline (Module-1) + Quality & Repair Engine (Module-2)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Ingest, process, repair, and export an audio/video file")
    run_p.add_argument("file", type=Path, help="Path to input media file (WAV, MP3, FLAC, M4A)")
    run_p.add_argument("--tgt", required=True, choices=["hi", "fr", "de", "es"], help="Target language code")
    run_p.add_argument(
        "--tts",
        default="mms_tts",
        choices=["mms_tts", "chatterbox"],
        help="TTS backend to use (default: mms_tts)",
    )
    run_p.add_argument("--skip-repair", action="store_true", help="Skip Module-2 repair loop (Module-1 only)")
    run_p.add_argument("--storage-dir", type=Path, default=Path("storage"), help="Storage directory (default: storage)")
    run_p.add_argument("--project-id", default=None, help="Optional fixed project ID")

    show_p = sub.add_parser("show", help="Show project details, segment metrics, and repair log as JSON")
    show_p.add_argument("project_id", help="Project ID")
    show_p.add_argument("--storage-dir", type=Path, default=Path("storage"), help="Storage directory")

    regen_p = sub.add_parser("regenerate", help="Regenerate a single segment in an existing project")
    regen_p.add_argument("project_id", help="Project ID")
    regen_p.add_argument("segment_idx", type=int, help="Zero-based segment index")
    regen_p.add_argument("--text", default=None, help="Optional updated target text before regenerating")
    regen_p.add_argument("--seed", type=int, default=None, help="Optional TTS seed")
    regen_p.add_argument("--storage-dir", type=Path, default=Path("storage"))

    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page; Hindi needs UTF-8
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)

    storage_dir: Path = args.storage_dir
    init_db(storage_dir)

    if args.command == "show":
        detail = get_project_detail(args.project_id, storage_dir)
        if not detail:
            print(json.dumps({"error": f"Project {args.project_id} not found"}), file=sys.stderr)
            return 1
        print(json.dumps(detail, indent=2, ensure_ascii=False))
        return 0

    registry = ModelRegistry()

    if args.command == "regenerate":
        try:
            seg_detail = regenerate_single_segment(
                project_id=args.project_id,
                segment_idx=args.segment_idx,
                new_tgt_text=args.text,
                seed=args.seed,
                registry=registry,
                storage_root=storage_dir,
            )
            print(json.dumps(seg_detail, indent=2, ensure_ascii=False))
            return 0
        finally:
            registry.close()

    if args.command == "run":
        input_path: Path = args.file
        try:
            created = create_project_from_file(
                src_path=input_path,
                target_lang=args.tgt,
                tts_provider=args.tts,
                storage_root=storage_dir,
                project_id=args.project_id,
            )
        except MediaValidationError as exc:
            print(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "error_code": exc.code,
                        "message": exc.reason,
                        "http_status": exc.http_status,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 2

        project_id = created["id"]
        print(f"[OMNICAST] Created project {project_id} ({input_path.name} -> {args.tgt}, tts={args.tts})")

        try:
            detail = run_pipeline_job(
                project_id=project_id,
                registry=registry,
                storage_root=storage_dir,
                enable_quality_repair=not args.skip_repair,
            )
        finally:
            registry.close()

        print(json.dumps(detail, indent=2, ensure_ascii=False))
        return 0 if detail.get("status") == "COMPLETED" else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
