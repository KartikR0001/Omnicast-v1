"""Module-1 Audio Timeline Reconstruction & SRT Subtitle Export (R4, R8)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass
class ReconstructedSegment:
    idx: int
    src_start_s: float
    src_end_s: float
    out_start_s: float
    out_end_s: float
    src_text: str
    tgt_text: str
    audio_path: str


def apply_fade(wav: np.ndarray, sr: int = 16000, fade_ms: float = 10.0) -> np.ndarray:
    """Apply short cosine fade-in and fade-out to prevent boundary click artifacts."""
    n_fade = int((sr * fade_ms) / 1000.0)
    if n_fade <= 1 or len(wav) < 2 * n_fade:
        return wav
    out = wav.copy()
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_fade, dtype=np.float32)))
    out[:n_fade] *= ramp
    out[-n_fade:] *= ramp[::-1]
    return out


def reconstruct_timeline(
    segments: list[dict],
    out_wav: Path | str,
    target_sr: int = 16000,
    min_gap_s: float = 0.08,
) -> tuple[float, list[ReconstructedSegment]]:
    """Stitch active segment audio files onto a master timeline without overlap (R4)."""
    placed: list[ReconstructedSegment] = []
    chunks: list[np.ndarray] = []
    cursor_s = 0.0

    for seg in sorted(segments, key=lambda s: int(s["idx"])):
        wav_path = Path(seg["audio_path"])
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing segment audio for reconstruction: {wav_path}")
        y, _ = librosa.load(wav_path, sr=target_sr, mono=True)
        y = apply_fade(y.astype(np.float32), sr=target_sr)
        seg_dur = len(y) / target_sr

        desired_start = float(seg.get("start_s", cursor_s))
        out_start = max(cursor_s, desired_start)
        gap_s = out_start - cursor_s
        if gap_s > 0.0:
            chunks.append(np.zeros(round(gap_s * target_sr), dtype=np.float32))
            cursor_s += gap_s

        chunks.append(y)
        out_end = cursor_s + seg_dur
        placed.append(
            ReconstructedSegment(
                idx=int(seg["idx"]),
                src_start_s=float(seg.get("start_s", 0.0)),
                src_end_s=float(seg.get("end_s", seg_dur)),
                out_start_s=round(out_start, 3),
                out_end_s=round(out_end, 3),
                src_text=str(seg.get("src_text", "")),
                tgt_text=str(seg.get("tgt_text", "")),
                audio_path=str(wav_path),
            )
                )
        cursor_s = out_end + min_gap_s

    full_audio = np.concatenate(chunks) if chunks else np.zeros(int(0.5 * target_sr), dtype=np.float32)
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, full_audio, target_sr)
    total_dur = len(full_audio) / target_sr
    return total_dur, placed


def format_srt_timestamp(seconds: float) -> str:
    """Format non-negative seconds as SubRip timestamp HH:MM:SS,mmm."""
    ms_total = max(0, round(seconds * 1000))
    hrs, rem = divmod(ms_total, 3600 * 1000)
    mins, rem = divmod(rem, 60 * 1000)
    secs, ms = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: list[ReconstructedSegment | dict], out_srt: Path | str) -> Path:
    """Write monotonic SubRip (.srt) subtitle file for reconstructed segments (R8)."""
    lines: list[str] = []
    prev_end = 0.0
    for i, seg in enumerate(segments, start=1):
        if isinstance(seg, ReconstructedSegment):
            s = max(prev_end, seg.out_start_s)
            e = max(s + 0.05, seg.out_end_s)
            txt = seg.tgt_text.strip() or seg.src_text.strip()
        else:
            s = max(prev_end, float(seg.get("out_start_s", seg.get("start_s", 0.0))))
            e = max(s + 0.05, float(seg.get("out_end_s", seg.get("end_s", s + 0.5))))
            txt = str(seg.get("tgt_text") or seg.get("src_text") or "").strip()
        prev_end = e
        lines.append(f"{i}")
        lines.append(f"{format_srt_timestamp(s)} --> {format_srt_timestamp(e)}")
        lines.append(txt)
        lines.append("")

    out_path = Path(out_srt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

