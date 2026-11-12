"""Module-2 Verified Selective Repair Engine (Part C.4, R6).

Executes policy decisions, writes versioned segment audio files (seg_NNN_v2.wav, seg_NNN_v3.wav)
without ever overwriting prior versions, verifies each candidate against pre-repair metrics,
rolls back on regression, and records an auditable RepairActionRecord for every step.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quality.detectors import (
    DEFAULT_QUALITY_CFG,
    QualityReportData,
    evaluate_segment,
    load_quality_config,
)
from quality.policy import RepairPolicy, verify_candidate


@dataclass
class RepairActionRecord:
    segment_idx: int
    attempt: int
    action: str
    reason: str
    accepted: bool
    verification_note: str
    candidate_version: int
    active_version_after: int
    before_metrics: dict[str, Any]
    after_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RepairEngine:
    def __init__(
        self,
        eval_asr,
        tts_fn: Callable[[str, str, str, str, int], Any],
        mt_fn: Callable[[str, str, str, int], str] | None = None,
        config_path: Path | str = DEFAULT_QUALITY_CFG,
    ):
        self.eval_asr = eval_asr
        self.tts_fn = tts_fn
        self.mt_fn = mt_fn
        self.config_path = Path(config_path)
        p_cfg = load_quality_config(self.config_path).get("policy", {})
        self.max_attempts = int(p_cfg.get("max_attempts", 2))
        self.seed_base = int(p_cfg.get("seed_base", 0))
        self.seed_stride = int(p_cfg.get("seed_stride", 101))
        self.policy = RepairPolicy(max_attempts=self.max_attempts)

    def evaluate(
        self,
        audio_path: Path | str,
        src_duration_s: float,
        tgt_text: str,
        target_lang: str,
        version: int,
        src_text: str | None = None,
    ) -> QualityReportData:
        return evaluate_segment(
            audio_path=audio_path,
            src_duration_s=src_duration_s,
            tgt_text=tgt_text,
            eval_asr=self.eval_asr,
            target_lang=target_lang,
            version=version,
            src_text=src_text,
            config_path=self.config_path,
        )

    def run_segment_repair(
        self,
        segment_idx: int,
        src_text: str,
        tgt_text: str,
        target_lang: str,
        src_wav_path: str,
        src_duration_s: float,
        segments_dir: Path | str,
        initial_report: QualityReportData,
        initial_version: int = 1,
    ) -> tuple[int, str, str, QualityReportData, list[QualityReportData], list[RepairActionRecord]]:
        """Run up to max_attempts verified repairs on a single segment.

        Returns:
            (active_version, active_audio_path, final_status, active_report, all_reports, action_log)
        """
        seg_dir = Path(segments_dir)
        active_ver = initial_version
        highest_ver = initial_version
        active_report = initial_report
        current_tgt_text = tgt_text
        all_reports = [initial_report]
        actions: list[RepairActionRecord] = []

        if not active_report.flags:
            return active_ver, active_report.audio_path, "ACCEPTED", active_report, all_reports, actions

        attempts = 0
        while active_report.flags and attempts < self.max_attempts:
            decision = self.policy.decide(active_report, attempts_so_far=attempts)
            if decision.action in ("ACCEPT", "HUMAN_REVIEW"):
                break

            attempts += 1
            highest_ver += 1
            cand_ver = highest_ver
            cand_wav = seg_dir / f"seg_{segment_idx:03d}_v{cand_ver}.wav"
            if cand_wav.exists():
                raise FileExistsError(f"Refusing to overwrite existing versioned audio: {cand_wav}")

            seed = self.seed_base + attempts * self.seed_stride + segment_idx
            cand_tgt_text = current_tgt_text

            if decision.action == "RETRANSLATE" and self.mt_fn is not None:
                cand_tgt_text = self.mt_fn(src_text, "en", target_lang, attempts)

            # Synthesize candidate into new versioned file
            self.tts_fn(cand_tgt_text, target_lang, src_wav_path, str(cand_wav), seed)

            cand_report = self.evaluate(
                audio_path=cand_wav,
                src_duration_s=src_duration_s,
                tgt_text=cand_tgt_text,
                target_lang=target_lang,
                version=cand_ver,
                src_text=src_text,
            )
            all_reports.append(cand_report)

            accepted, note = verify_candidate(active_report, cand_report)
            before_snap = {
                "version": active_report.version,
                "cer": active_report.cer,
                "duration_ratio": active_report.duration_ratio,
                "clipping_fraction": active_report.clipping_fraction,
                "silence_ratio": active_report.silence_ratio,
                "flags": list(active_report.flags),
            }
            after_snap = {
                "version": cand_report.version,
                "cer": cand_report.cer,
                "duration_ratio": cand_report.duration_ratio,
                "clipping_fraction": cand_report.clipping_fraction,
                "silence_ratio": cand_report.silence_ratio,
                "flags": list(cand_report.flags),
            }

            if accepted:
                active_ver = cand_ver
                active_report = cand_report
                current_tgt_text = cand_tgt_text

            actions.append(
                RepairActionRecord(
                    segment_idx=segment_idx,
                    attempt=attempts,
                    action=decision.action,
                    reason=decision.reason,
                    accepted=accepted,
                    verification_note=note,
                    candidate_version=cand_ver,
                    active_version_after=active_ver,
                    before_metrics=before_snap,
                    after_metrics=after_snap,
                )
            )

        if active_report.flags:
            esc = self.policy.decide(active_report, attempts_so_far=attempts)
            actions.append(
                RepairActionRecord(
                    segment_idx=segment_idx,
                    attempt=attempts,
                    action="HUMAN_REVIEW",
                    reason=esc.reason,
                    accepted=False,
                    verification_note="escalated_after_max_attempts",
                    candidate_version=active_ver,
                    active_version_after=active_ver,
                    before_metrics={
                        "version": active_report.version,
                        "cer": active_report.cer,
                        "flags": list(active_report.flags),
                    },
                    after_metrics={
                        "version": active_report.version,
                        "cer": active_report.cer,
                        "flags": list(active_report.flags),
                    },
                )
            )
            final_status = "HUMAN_REVIEW"
        else:
            final_status = "REPAIRED"

        return active_ver, active_report.audio_path, final_status, active_report, all_reports, actions

