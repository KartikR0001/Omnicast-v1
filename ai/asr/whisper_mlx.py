"""Whisper ASR via mlx-whisper (Apple Silicon). In-loop ASR family for OMNICAST."""
from __future__ import annotations

import time

import numpy as np

from ai.base import ComponentResult

DEFAULT_REPO = "mlx-community/whisper-large-v3-turbo"


class WhisperMLX:
    def __init__(self, repo: str = DEFAULT_REPO):
        import mlx_whisper  # noqa: F401  (import check)
        self.repo = repo

    def transcribe(self, audio_16k: np.ndarray, language: str | None = None, word_timestamps: bool = False) -> ComponentResult:
        import mlx_whisper
        t0 = time.perf_counter()
        r = mlx_whisper.transcribe(audio_16k.astype(np.float32), path_or_hf_repo=self.repo, language=language,
                                   word_timestamps=word_timestamps, temperature=0.0, condition_on_previous_text=False,
                                   verbose=None)
        dt = time.perf_counter() - t0
        segs = r.get("segments", [])
        extra = {
            "language": r.get("language"),
            "avg_logprob": [s.get("avg_logprob") for s in segs],
            "no_speech_prob": [s.get("no_speech_prob") for s in segs],
            "compression_ratio": [s.get("compression_ratio") for s in segs],
            "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs],
        }
        return ComponentResult(r["text"].strip(), self.repo, dt, extra)
