"""Unified Pipeline & Quality Execution Engine + Background JobWorker (app/worker.py).

Shared by both the FastAPI asynchronous worker (R10) and the headless CLI (cli.py).
Updates SQLite state transitions in real time and records explicit error reasons on failure (R9).
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

import soundfile as sf

from ai.registry import ModelRegistry
from app.db import db_conn, get_storage_root, row_to_dict, utc_now
from app.models import get_project_detail
from pipeline.ingest import ingest_file
from pipeline.reconstruct import reconstruct_timeline, write_srt
from pipeline.segmenter import segment_and_transcribe
from quality.policy import verify_candidate
from quality.repair import RepairEngine


def _insert_quality_report(conn, project_id: str, segment_idx: int, rep) -> None:
    conn.execute(
        """
        INSERT INTO quality_reports (
            project_id, segment_idx, version, audio_path, src_duration_s, gen_duration_s,
            cer, duration_ratio, clipping_fraction, silence_ratio, empty_audio,
            bt_chrf, eval_transcript, flags_json, created_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            segment_idx,
            rep.version,
            rep.audio_path,
            rep.src_duration_s,
            rep.gen_duration_s,
            rep.cer,
            rep.duration_ratio,
            rep.clipping_fraction,
            rep.silence_ratio,
            1 if rep.empty_audio else 0,
            rep.bt_chrf,
            rep.eval_transcript,
            json.dumps(rep.flags),
            utc_now(),
        ),
    )


def _insert_repair_action(conn, project_id: str, act) -> None:
    conn.execute(
        """
        INSERT INTO repair_actions (
            project_id, segment_idx, attempt, action, reason, accepted,
            verification_note, candidate_version, active_version_after,
            before_json, after_json, created_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            act.segment_idx,
            act.attempt,
            act.action,
            act.reason,
            1 if act.accepted else 0,
            act.verification_note,
            act.candidate_version,
            act.active_version_after,
            json.dumps(act.before_metrics),
            json.dumps(act.after_metrics),
            utc_now(),
        ),
    )


def create_project_from_file(
    src_path: Path | str,
    target_lang: str,
    name: str | None = None,
    tts_provider: str = "mms_tts",
    storage_root: Path | str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Validate media (R1), copy read-only original to D1 (R11), and create QUEUED project row."""
    if target_lang not in ("hi", "fr", "de", "es"):
        raise ValueError(f"Unsupported target language '{target_lang}'. Allowed: hi, fr, de, es")

    root = get_storage_root(storage_root)
    pid = project_id or uuid.uuid4().hex[:12]
    src_p = Path(src_path)
    meta, orig_dest, work_wav, _ = ingest_file(src_p, root, pid)
    now = utc_now()
    proj_name = (name or src_p.stem).strip() or pid

    with db_conn(root) as conn:
        conn.execute(
            """
            INSERT INTO projects (
                id, name, source_filename, original_path, work_wav_path,
                target_lang, tts_provider, status, stage, error_reason,
                duration_s, sample_rate, channels, size_bytes, sha256,
                created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', 'INGESTED', NULL, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                proj_name,
                src_p.name,
                str(orig_dest),
                str(work_wav),
                target_lang,
                tts_provider,
                meta.duration_s,
                meta.sample_rate,
                meta.channels,
                meta.size_bytes,
                meta.sha256,
                now,
                now,
            ),
        )
    return get_project_detail(pid, root)


def _set_project_state(root: Path, project_id: str, status: str, stage: str, error_reason: str | None = None) -> None:
    with db_conn(root) as conn:
        conn.execute(
            "UPDATE projects SET status = ?, stage = ?, error_reason = ?, updated_utc = ? WHERE id = ?",
            (status, stage, error_reason, utc_now(), project_id),
        )


def _rebuild_project_exports(root: Path, project_id: str, target_lang: str) -> tuple[Path, Path, float]:
    with db_conn(root) as conn:
        rows = conn.execute(
            "SELECT * FROM segments WHERE project_id = ? ORDER BY idx ASC", (project_id,)
        ).fetchall()
    seg_dicts = [
        {
            "idx": r["idx"],
            "start_s": r["start_s"],
            "end_s": r["end_s"],
            "src_text": r["src_text"],
            "tgt_text": r["tgt_text"],
            "audio_path": r["active_audio_path"],
        }
        for r in rows
    ]
    exp_dir = root / "exports" / project_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_wav = exp_dir / f"{project_id}_{target_lang}.wav"
    out_srt = exp_dir / f"{project_id}_{target_lang}.srt"
    out_dur, placed = reconstruct_timeline(seg_dicts, out_wav)
    write_srt(placed, out_srt)

    with db_conn(root) as conn:
        for p in placed:
            conn.execute(
                "UPDATE segments SET out_start_s = ?, out_end_s = ? WHERE project_id = ? AND idx = ?",
                (p.out_start_s, p.out_end_s, project_id, p.idx),
            )
        conn.execute(
            """
            UPDATE projects
            SET out_wav_path = ?, out_srt_path = ?, out_duration_s = ?, updated_utc = ?
            WHERE id = ?
            """,
            (str(out_wav), str(out_srt), out_dur, utc_now(), project_id),
        )
    return out_wav, out_srt, out_dur


def run_pipeline_job(
    project_id: str,
    registry: ModelRegistry | None = None,
    storage_root: Path | str | None = None,
    enable_quality_repair: bool = True,
) -> dict[str, Any]:
    """Execute Module-1 and Module-2 end-to-end for a queued project."""
    root = get_storage_root(storage_root)
    own_registry = registry is None
    reg = registry or ModelRegistry()

    try:
        with db_conn(root) as conn:
            prow = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not prow:
                raise KeyError(f"Project {project_id} not found")
            proj = row_to_dict(prow)

        target_lang = proj["target_lang"]
        tts_provider = proj["tts_provider"]
        work_wav = Path(proj["work_wav_path"])
        seg_dir = root / "segments" / project_id
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: ASR & Segmentation (R2)
        _set_project_state(root, project_id, "RUNNING", "ASR_SEGMENTATION")
        audio_16k, sr = sf.read(work_wav, dtype="float32")
        asr_model = reg.get_asr()
        t_segs = segment_and_transcribe(audio_16k, asr_model, seg_dir, sr=sr, language="en")
        if not t_segs:
            raise RuntimeError("ASR produced 0 segments from input audio")

        if reg.cfg.get("memory", {}).get("unload_inloop_asr_before_tts", True) and tts_provider == "chatterbox":
            reg.unload_asr()

        # Stage 2: MT & Initial TTS Synthesis v1 (R3, R4)
        _set_project_state(root, project_id, "RUNNING", "TRANSLATION_AND_TTS")
        mt_model = reg.get_mt()
        tts_model = reg.get_tts(tts_provider)

        for seg in t_segs:
            mt_res = mt_model.translate(seg.src_text, "en", target_lang)
            tgt_text = (mt_res.output or "").strip()
            v1_wav = seg_dir / f"seg_{seg.idx:03d}_v1.wav"
            tts_res = tts_model.synthesize(
                tgt_text,
                target_lang,
                prompt_wav=seg.src_wav_path,
                out_wav=str(v1_wav),
                seed=0,
            )
            with db_conn(root) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO segments (
                        project_id, idx, start_s, end_s, src_text, tgt_text,
                        src_wav_path, active_version, active_audio_path, status,
                        asr_seconds, mt_seconds, tts_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 'SYNTHESIZED', ?, ?, ?)
                    """,
                    (
                        project_id,
                        seg.idx,
                        seg.start_s,
                        seg.end_s,
                        seg.src_text,
                        tgt_text,
                        seg.src_wav_path,
                        str(v1_wav),
                        seg.asr_seconds,
                        mt_res.seconds,
                        tts_res.seconds,
                    ),
                )

        # Stage 3: Quality Detection, Policy v0 & Verified Selective Repair (R5, R6)
        if enable_quality_repair:
            _set_project_state(root, project_id, "RUNNING", "QUALITY_AND_REPAIR")
            eval_asr = reg.get_eval_asr()

            def _tts_call(text: str, lang: str, prompt_wav: str, out_wav: str, seed: int):
                rate = 1.0 if (seed % 2 == 0) else 0.94
                if hasattr(tts_model, "synthesize"):
                    try:
                        return tts_model.synthesize(text, lang, prompt_wav, out_wav, seed=seed, speaking_rate=rate)
                    except TypeError:
                        return tts_model.synthesize(text, lang, prompt_wav, out_wav, seed=seed)
                raise RuntimeError("Invalid TTS model")

            def _mt_call(src_txt: str, src_l: str, tgt_l: str, attempt_num: int) -> str:
                beams = 4 + attempt_num * 2
                return mt_model.translate(src_txt, src_l, tgt_l, num_beams=beams).output

            engine = RepairEngine(eval_asr=eval_asr, tts_fn=_tts_call, mt_fn=_mt_call)

            with db_conn(root) as conn:
                db_segs = conn.execute(
                    "SELECT * FROM segments WHERE project_id = ? ORDER BY idx ASC", (project_id,)
                ).fetchall()

            for srow in db_segs:
                sidx = int(srow["idx"])
                src_dur = max(0.01, float(srow["end_s"]) - float(srow["start_s"]))
                init_rep = engine.evaluate(
                    audio_path=srow["active_audio_path"],
                    src_duration_s=src_dur,
                    tgt_text=srow["tgt_text"],
                    target_lang=target_lang,
                    version=1,
                    src_text=srow["src_text"],
                )
                active_ver, active_wav, final_status, _, all_reps, actions = engine.run_segment_repair(
                    segment_idx=sidx,
                    src_text=srow["src_text"],
                    tgt_text=srow["tgt_text"],
                    target_lang=target_lang,
                    src_wav_path=srow["src_wav_path"],
                    src_duration_s=src_dur,
                    segments_dir=seg_dir,
                    initial_report=init_rep,
                    initial_version=1,
                )
                with db_conn(root) as conn:
                    for rep in all_reps:
                        _insert_quality_report(conn, project_id, sidx, rep)
                    for act in actions:
                        _insert_repair_action(conn, project_id, act)
                    conn.execute(
                        """
                        UPDATE segments
                        SET active_version = ?, active_audio_path = ?, status = ?
                        WHERE project_id = ? AND idx = ?
                        """,
                        (active_ver, active_wav, final_status, project_id, sidx),
                    )
        else:
            with db_conn(root) as conn:
                conn.execute("UPDATE segments SET status = 'ACCEPTED' WHERE project_id = ?", (project_id,))

        # Stage 4: Timeline Reconstruction & SRT Export (R4, R8)
        _set_project_state(root, project_id, "RUNNING", "RECONSTRUCTION_AND_EXPORT")
        _rebuild_project_exports(root, project_id, target_lang)
        _set_project_state(root, project_id, "COMPLETED", "COMPLETED", None)
        return get_project_detail(project_id, root)

    except Exception as e:
        _set_project_state(root, project_id, "FAILED", "FAILED", f"{type(e).__name__}: {e}")
        raise
    finally:
        if own_registry:
            reg.close()


def regenerate_single_segment(
    project_id: str,
    segment_idx: int,
    new_tgt_text: str | None = None,
    seed: int | None = None,
    registry: ModelRegistry | None = None,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Re-synthesize a single segment on user request (R7), verify quality, and update exports."""
    root = get_storage_root(storage_root)
    own_registry = registry is None
    reg = registry or ModelRegistry()
    try:
        with db_conn(root) as conn:
            prow = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not prow:
                raise KeyError(f"Project {project_id} not found")
            srow = conn.execute(
                "SELECT * FROM segments WHERE project_id = ? AND idx = ?", (project_id, segment_idx)
            ).fetchone()
            if not srow:
                raise KeyError(f"Segment {segment_idx} not found in project {project_id}")
            max_ver_row = conn.execute(
                "SELECT MAX(version) AS mv FROM quality_reports WHERE project_id = ? AND segment_idx = ?",
                (project_id, segment_idx),
            ).fetchone()
            prev_rep_row = conn.execute(
                """
                SELECT * FROM quality_reports
                WHERE project_id = ? AND segment_idx = ? AND version = ?
                ORDER BY id DESC LIMIT 1
                """,
                (project_id, segment_idx, srow["active_version"]),
            ).fetchone()
            act_count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM repair_actions WHERE project_id = ? AND segment_idx = ?",
                (project_id, segment_idx),
            ).fetchone()

        target_lang = prow["target_lang"]
        tts_provider = prow["tts_provider"]
        text_edited = new_tgt_text is not None or (srow["status"] == "TEXT_EDITED")
        tgt_text = (new_tgt_text if new_tgt_text is not None else srow["tgt_text"]).strip()
        if not tgt_text:
            raise ValueError("Target translation text cannot be empty")

        cur_active_ver = int(srow["active_version"])
        highest_ver = max(cur_active_ver, int(max_ver_row["mv"] or cur_active_ver))
        cand_ver = highest_ver + 1
        attempt_num = int(act_count_row["cnt"] or 0) + 1
        chosen_seed = seed if seed is not None else (200 + cand_ver * 37 + segment_idx)

        seg_dir = root / "segments" / project_id
        seg_dir.mkdir(parents=True, exist_ok=True)
        cand_wav = seg_dir / f"seg_{segment_idx:03d}_v{cand_ver}.wav"

        tts_model = reg.get_tts(tts_provider)
        eval_asr = reg.get_eval_asr()
        tts_res = tts_model.synthesize(
            tgt_text,
            target_lang,
            prompt_wav=srow["src_wav_path"],
            out_wav=str(cand_wav),
            seed=chosen_seed,
        )

        src_dur = max(0.01, float(srow["end_s"]) - float(srow["start_s"]))
        from quality.detectors import QualityReportData, evaluate_segment
        from quality.repair import RepairActionRecord

        cand_rep = evaluate_segment(
            audio_path=cand_wav,
            src_duration_s=src_dur,
            tgt_text=tgt_text,
            eval_asr=eval_asr,
            target_lang=target_lang,
            version=cand_ver,
            src_text=srow["src_text"],
        )

        if prev_rep_row:
            prev_flags = json.loads(prev_rep_row["flags_json"])
            before_rep = QualityReportData(
                version=int(prev_rep_row["version"]),
                audio_path=prev_rep_row["audio_path"],
                src_duration_s=prev_rep_row["src_duration_s"],
                gen_duration_s=prev_rep_row["gen_duration_s"],
                cer=prev_rep_row["cer"],
                duration_ratio=prev_rep_row["duration_ratio"],
                clipping_fraction=prev_rep_row["clipping_fraction"],
                silence_ratio=prev_rep_row["silence_ratio"],
                empty_audio=bool(prev_rep_row["empty_audio"]),
                flags=prev_flags,
            )
        else:
            before_rep = cand_rep

        if text_edited:
            # Explicit human text edit is accepted unless candidate audio is empty/corrupt
            accepted = not cand_rep.empty_audio
            note = "user_edited_text_accepted" if accepted else "rejected_empty_audio"
        elif before_rep.flags:
            accepted, note = verify_candidate(before_rep, cand_rep)
        else:
            # Manual re-synthesis of an already clean segment: accept if no new flags and CER does not regress
            accepted = (not cand_rep.flags) and (
                before_rep.cer is None or (cand_rep.cer is not None and cand_rep.cer <= before_rep.cer + 0.05)
            )
            note = "manual_regen_verified" if accepted else "manual_regen_rollback_worse_metrics"

        new_active_ver = cand_ver if accepted else cur_active_ver
        new_active_wav = str(cand_wav) if accepted else srow["active_audio_path"]
        new_tgt = tgt_text if accepted else srow["tgt_text"]
        active_flags = cand_rep.flags if accepted else before_rep.flags
        new_status = "REPAIRED" if not active_flags else "HUMAN_REVIEW"

        act_rec = RepairActionRecord(
            segment_idx=segment_idx,
            attempt=attempt_num,
            action="USER_EDIT_REGEN" if text_edited else "REGENERATE_TTS",
            reason="manual_review_trigger",
            accepted=accepted,
            verification_note=note,
            candidate_version=cand_ver,
            active_version_after=new_active_ver,
            before_metrics={
                "version": before_rep.version,
                "cer": before_rep.cer,
                "duration_ratio": before_rep.duration_ratio,
                "flags": before_rep.flags,
            },
            after_metrics={
                "version": cand_rep.version,
                "cer": cand_rep.cer,
                "duration_ratio": cand_rep.duration_ratio,
                "flags": cand_rep.flags,
            },
        )

        with db_conn(root) as conn:
            _insert_quality_report(conn, project_id, segment_idx, cand_rep)
            _insert_repair_action(conn, project_id, act_rec)
            conn.execute(
                """
                UPDATE segments
                SET tgt_text = ?, active_version = ?, active_audio_path = ?, status = ?, tts_seconds = ?
                WHERE project_id = ? AND idx = ?
                """,
                (new_tgt, new_active_ver, new_active_wav, new_status, tts_res.seconds, project_id, segment_idx),
            )

        _rebuild_project_exports(root, project_id, target_lang)
        detail = get_project_detail(project_id, root)
        return next(s for s in detail["segments"] if s["idx"] == segment_idx)
    finally:
        if own_registry:
            reg.close()


class JobWorker:
    """In-process background worker queue so uploads return HTTP 202 immediately (R10)."""

    def __init__(self, registry: ModelRegistry | None = None, storage_root: Path | str | None = None):
        self.registry = registry or ModelRegistry()
        self.storage_root = storage_root
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="omnicast-job-worker", daemon=True)
        self._thread.start()

    def submit(self, project_id: str) -> None:
        self._queue.put(project_id)

    def _loop(self) -> None:
        while True:
            pid = self._queue.get()
            if pid is None:
                self._queue.task_done()
                break
            try:
                run_pipeline_job(pid, registry=self.registry, storage_root=self.storage_root)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def wait_all(self, timeout: float = 120.0) -> None:
        self._queue.join()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self.registry.close()
