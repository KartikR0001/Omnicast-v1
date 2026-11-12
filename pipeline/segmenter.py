"""Module-1 Audio Segmentation & In-Loop ASR Transcription (R2).

Combines WhisperX-inspired min-energy Cut & Merge pre-chunking (for recordings > 30 s)
with WhisperMLX segment extraction and strict boundary normalization (monotonic,
non-overlapping, clamped to [0, total_duration_s]).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class TranscriptSegment:
    idx: int
    start_s: float
    end_s: float
    src_text: str
    src_wav_path: str = ""
    asr_model_id: str = ""
    asr_seconds: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def find_min_energy_cut(
    audio_16k: np.ndarray,
    search_start_s: float,
    search_end_s: float,
    sr: int = 16000,
    frame_ms: int = 20,
) -> float:
    """Find the time (in seconds) of minimum RMS energy within [search_start_s, search_end_s]."""
    i0 = max(0, round(search_start_s * sr))
    i1 = min(len(audio_16k), round(search_end_s * sr))
    if i1 <= i0:
        return search_start_s
    frame_len = max(1, (sr * frame_ms) // 1000)
    window = audio_16k[i0:i1]
    n_frames = len(window) // frame_len
    if n_frames <= 1:
        return (search_start_s + search_end_s) / 2.0
    trimmed = window[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
    min_idx = int(np.argmin(rms))
    return (i0 + min_idx * frame_len) / sr


def cut_and_merge_windows(
    audio_16k: np.ndarray,
    sr: int = 16000,
    max_window_s: float = 28.0,
) -> list[tuple[float, float]]:
    """Split audio longer than max_window_s at minimum-energy points in [0.5*max_window_s, max_window_s]."""
    total_s = len(audio_16k) / sr
    if total_s <= 0.0:
        return []
    if total_s <= max_window_s:
        return [(0.0, total_s)]

    windows: list[tuple[float, float]] = []
    cur = 0.0
    while total_s - cur > max_window_s:
        cut_s = find_min_energy_cut(
            audio_16k,
            cur + 0.5 * max_window_s,
            cur + max_window_s,
            sr=sr,
        )
        if cut_s <= cur + 0.1:
            cut_s = cur + max_window_s
        windows.append((round(cur, 3), round(cut_s, 3)))
        cur = cut_s
    if total_s - cur >= 0.25:
        windows.append((round(cur, 3), round(total_s, 3)))
    elif windows:
        prev_start, _ = windows[-1]
        windows[-1] = (prev_start, round(total_s, 3))
    return windows


def normalize_segment_boundaries(
    raw_segments: list[dict],
    total_duration_s: float,
    min_seg_s: float = 0.25,
    max_seg_s: float = 30.0,
) -> list[TranscriptSegment]:
    """Ensure segment boundaries are within [0, total_duration_s], monotonic, non-overlapping, and non-empty."""
    if total_duration_s <= 0.0:
        return []

    cleaned: list[TranscriptSegment] = []
    cursor = 0.0
    for item in raw_segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        s = float(item.get("start", 0.0))
        e = float(item.get("end", total_duration_s))
        s = max(cursor, max(0.0, min(s, total_duration_s)))
        e = max(s, min(e, total_duration_s))
        if e - s < min_seg_s:
            e = min(total_duration_s, s + max(min_seg_s, 0.5))
        if e - s < 0.05:
            continue

        # Sub-divide any segment exceeding max_seg_s proportionally by word count
        if e - s > max_seg_s:
            words = text.split()
            n_parts = int(np.ceil((e - s) / max_seg_s))
            part_dur = (e - s) / n_parts
            words_per_part = max(1, int(np.ceil(len(words) / n_parts)))
            for p_idx in range(n_parts):
                w_slice = words[p_idx * words_per_part : (p_idx + 1) * words_per_part]
                if not w_slice:
                    continue
                ps = round(s + p_idx * part_dur, 3)
                pe = round(min(e, s + (p_idx + 1) * part_dur), 3)
                if pe > ps:
                    cleaned.append(
                        TranscriptSegment(
                            idx=len(cleaned),
                            start_s=ps,
                            end_s=pe,
                            src_text=" ".join(w_slice),
                            asr_model_id=str(item.get("model_id", "")),
                            asr_seconds=float(item.get("seconds", 0.0)) / n_parts,
                        )
                    )
            cursor = e
        else:
            cleaned.append(
                TranscriptSegment(
                    idx=len(cleaned),
                    start_s=round(s, 3),
                    end_s=round(e, 3),
                    src_text=text,
                    asr_model_id=str(item.get("model_id", "")),
                    asr_seconds=float(item.get("seconds", 0.0)),
                )
            )
            cursor = e

    return cleaned


def segment_and_transcribe(
    audio_16k: np.ndarray,
    asr_model,
    segments_dir: Path | str,
    sr: int = 16000,
    language: str = "en",
) -> list[TranscriptSegment]:
    """Run VAD Cut & Merge + in-loop ASR and write per-segment source WAV slices (R2)."""
    total_dur = len(audio_16k) / sr
    windows = cut_and_merge_windows(audio_16k, sr=sr, max_window_s=28.0)
    raw_segs: list[dict] = []

    for w_start, w_end in windows:
        i0 = round(w_start * sr)
        i1 = round(w_end * sr)
        chunk = audio_16k[i0:i1]
        if len(chunk) == 0:
            continue
        res = asr_model.transcribe(chunk, language=language)
        w_segs = res.extra.get("segments") if isinstance(res.extra, dict) else None
        if w_segs:
            n_s = max(1, len(w_segs))
            for s in w_segs:
                raw_segs.append(
                    {
                        "start": w_start + float(s.get("start", 0.0)),
                        "end": min(w_end, w_start + float(s.get("end", w_end - w_start))),
                        "text": s.get("text", ""),
                        "model_id": res.model_id,
                        "seconds": res.seconds / n_s,
                    }
                )
        elif (res.output or "").strip():
            raw_segs.append(
                {
                    "start": w_start,
                    "end": w_end,
                    "text": res.output,
                    "model_id": res.model_id,
                    "seconds": res.seconds,
                }
            )

    segs = normalize_segment_boundaries(raw_segments=raw_segs, total_duration_s=total_dur)
    if not segs and total_dur > 0:
        # Fallback if ASR produced a top-level string that got filtered
        res = asr_model.transcribe(audio_16k, language=language)
        text = (res.output or "").strip()
        if text:
            segs = [
                TranscriptSegment(
                    idx=0,
                    start_s=0.0,
                    end_s=round(total_dur, 3),
                    src_text=text,
                    asr_model_id=res.model_id,
                    asr_seconds=res.seconds,
                )
            ]

    seg_dir = Path(segments_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    for seg in segs:
        i0 = max(0, round(seg.start_s * sr))
        i1 = min(len(audio_16k), round(seg.end_s * sr))
        clip = audio_16k[i0:i1]
        if len(clip) == 0:
            clip = np.zeros(int(0.25 * sr), dtype=np.float32)
        src_wav = seg_dir / f"seg_{seg.idx:03d}_src.wav"
        sf.write(src_wav, clip, sr)
        seg.src_wav_path = str(src_wav)

    return segs

