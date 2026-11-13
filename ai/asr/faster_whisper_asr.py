"""Whisper ASR via faster-whisper (CTranslate2). Runs on Windows, Linux and Intel Macs (CPU or NVIDIA GPU).

Same interface and output as WhisperMLX, so the rest of the pipeline does not change. Used automatically
when mlx-whisper is not available (anything that is not an Apple Silicon Mac).
"""
from __future__ import annotations

import time

import numpy as np

from ai.base import ComponentResult

DEFAULT_MODEL = "small"


class FasterWhisper:
    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu", compute_type: str = "auto"):
        if device == "auto":
            device = "cpu"
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:
                pass
        self.model_name = model
        self._compute_type = compute_type
        self._load(device)

    def _load(self, device: str) -> None:
        from faster_whisper import WhisperModel

        compute_type = self._compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self.device = device
        self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    def _run(self, audio: np.ndarray, language: str | None, word_timestamps: bool):
        seg_iter, info = self.model.transcribe(
            audio, language=language, word_timestamps=word_timestamps, temperature=0.0,
            condition_on_previous_text=False, beam_size=5, vad_filter=False,
        )
        return list(seg_iter), info  # the generator does the actual decoding

    def transcribe(self, audio_16k: np.ndarray, language: str | None = None, word_timestamps: bool = False) -> ComponentResult:
        t0 = time.perf_counter()
        audio = np.ascontiguousarray(audio_16k, dtype=np.float32)
        try:
            segs, info = self._run(audio, language, word_timestamps)
        except (RuntimeError, OSError) as exc:
            if self.device == "cpu":
                raise
            # GPU present but CUDA/cuDNN libraries missing (common on Windows): retry on the CPU
            print(f"[OMNICAST] GPU speech-to-text failed ({exc}); switching to CPU.", flush=True)
            self._load("cpu")
            segs, info = self._run(audio, language, word_timestamps)
        dt = time.perf_counter() - t0
        text = "".join(s.text for s in segs).strip()
        extra = {
            "language": info.language,
            "avg_logprob": [s.avg_logprob for s in segs],
            "no_speech_prob": [s.no_speech_prob for s in segs],
            "compression_ratio": [s.compression_ratio for s in segs],
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segs],
        }
        return ComponentResult(text, f"faster-whisper/{self.model_name} ({self.device})", dt, extra)
