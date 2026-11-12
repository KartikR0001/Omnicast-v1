"""Module-2 Per-Segment Quality Detectors D1–D4 (R5).

All thresholds are read from config/quality.yaml.
If any metric cannot be computed, it is stored as None (NULL in SQLite) — never imputed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import yaml

from orchestration.quality.text_metrics import cer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUALITY_CFG = ROOT / "config/quality.yaml"


def load_quality_config(path: Path | str = DEFAULT_QUALITY_CFG) -> dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


@dataclass
class QualityReportData:
    version: int
    audio_path: str
    src_duration_s: float | None
    gen_duration_s: float | None
    cer: float | None
    duration_ratio: float | None
    clipping_fraction: float | None
    silence_ratio: float | None
    empty_audio: bool
    bt_chrf: float | None = None
    eval_transcript: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_acoustic_metrics(
    samples: np.ndarray,
    sr: int,
    frame_ms: int = 20,
    clip_level: float = 0.999,
    silence_rms: float = 1e-4,
) -> tuple[float, float, float | None, float | None]:
    """Return (gen_duration_s, peak, clipping_fraction, silence_ratio) on a 1D waveform."""
    if sr <= 0 or samples.size == 0:
        return 0.0, 0.0, None, None
    x = samples.astype(np.float32).reshape(-1)
    if not np.isfinite(x).all():
        return len(x) / sr, float("nan"), 1.0, 1.0

    dur = len(x) / sr
    peak = float(np.max(np.abs(x)))
    clip_frac = float(np.mean(np.abs(x) >= clip_level))

    frame_len = max(1, (sr * frame_ms) // 1000)
    n_frames = len(x) // frame_len
    if n_frames == 0:
        rms = float(np.sqrt(np.mean(x ** 2)))
        sil_ratio = 1.0 if rms < silence_rms else 0.0
    else:
        framed = x[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(framed ** 2, axis=1))
        sil_ratio = float(np.mean(frame_rms < silence_rms))

    return dur, peak, clip_frac, sil_ratio


def evaluate_segment(
    audio_path: Path | str,
    src_duration_s: float | None,
    tgt_text: str,
    eval_asr=None,
    target_lang: str = "hi",
    version: int = 1,
    src_text: str | None = None,
    back_translated_text: str | None = None,
    config_path: Path | str = DEFAULT_QUALITY_CFG,
) -> QualityReportData:
    """Run D1 (round-trip CER), D2 (duration ratio), D3 (acoustic) and D4 (empty-translation check).

    D4 note (audit F-11): in Phase-1 D4 only flags an empty translation. The back-translation chrF branch runs
    only if a caller passes `back_translated_text`; no production caller does, so `bt_chrf` is always NULL.
    """
    cfg = load_quality_config(config_path).get("detectors", {})
    cer_max = float(cfg.get("cer_max", 0.35))
    r_lo = float(cfg.get("duration_ratio_min", 0.45))
    r_hi = float(cfg.get("duration_ratio_max", 2.20))
    clip_max = float(cfg.get("clipping_fraction_max", 0.01))
    sil_max = float(cfg.get("silence_ratio_max", 0.85))
    min_audio_s = float(cfg.get("min_audio_seconds", 0.15))

    p = Path(audio_path)
    flags: list[str] = []

    if not p.exists() or p.stat().st_size == 0:
        flags.append("acoustic")
        if not (tgt_text or "").strip():
            flags.append("translation")
        return QualityReportData(
            version=version,
            audio_path=str(p),
            src_duration_s=src_duration_s,
            gen_duration_s=0.0,
            cer=None,
            duration_ratio=None,
            clipping_fraction=None,
            silence_ratio=None,
            empty_audio=True,
            bt_chrf=None,
            eval_transcript=None,
            flags=flags,
        )

    try:
        x, sr = sf.read(p, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
    except Exception:
        flags.append("acoustic")
        return QualityReportData(
            version=version,
            audio_path=str(p),
            src_duration_s=src_duration_s,
            gen_duration_s=None,
            cer=None,
            duration_ratio=None,
            clipping_fraction=None,
            silence_ratio=None,
            empty_audio=True,
            bt_chrf=None,
            eval_transcript=None,
            flags=flags,
        )

    gen_dur, _, clip_frac, sil_ratio = compute_acoustic_metrics(x, sr)
    empty_audio = gen_dur < min_audio_s

    # D3: Acoustic flag
    if empty_audio or (clip_frac is not None and clip_frac > clip_max) or (sil_ratio is not None and sil_ratio > sil_max):
        flags.append("acoustic")

    # D2: Duration ratio
    dur_ratio: float | None = None
    if src_duration_s is not None and src_duration_s > 0.0 and gen_dur > 0.0:
        dur_ratio = gen_dur / src_duration_s
        if dur_ratio < r_lo or dur_ratio > r_hi:
            flags.append("timing")

    # D1: Round-trip CER via held-out MMS-CTC
    rt_cer: float | None = None
    eval_transcript: str | None = None
    if not empty_audio and eval_asr is not None and (tgt_text or "").strip():
        y_16k = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        asr_res = eval_asr.transcribe(y_16k, target_lang)
        eval_transcript = asr_res.output
        rt_cer = cer(tgt_text, eval_transcript)
        if rt_cer is not None and rt_cer > cer_max:
            flags.append("content")

    # D4: empty-translation check (back-translation chrF branch unused in Phase-1 production paths)
    bt_chrf: float | None = None
    if not (tgt_text or "").strip():
        flags.append("translation")
    elif back_translated_text is not None and (src_text or "").strip():
        import sacrebleu
        bt_chrf = float(sacrebleu.sentence_chrf(back_translated_text, [src_text]).score)
        bt_min = float(cfg.get("bt_chrf_min", 25.0))
        if bt_chrf < bt_min:
            flags.append("translation")

    return QualityReportData(
        version=version,
        audio_path=str(p),
        src_duration_s=src_duration_s,
        gen_duration_s=round(gen_dur, 4),
        cer=round(rt_cer, 4) if rt_cer is not None else None,
        duration_ratio=round(dur_ratio, 4) if dur_ratio is not None else None,
        clipping_fraction=round(clip_frac, 4) if clip_frac is not None else None,
        silence_ratio=round(sil_ratio, 4) if sil_ratio is not None else None,
        empty_audio=empty_audio,
        bt_chrf=round(bt_chrf, 2) if bt_chrf is not None else None,
        eval_transcript=eval_transcript,
        flags=flags,
    )

