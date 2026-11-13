"""Config-driven ModelRegistry (R12).

Loads each AI wrapper lazily (at most once per process instead of re-instantiating per segment)
and supports unloading in-loop ASR before heavy TTS synthesis to stay within the 18 GB unified
memory budget on Apple Silicon.
"""
from __future__ import annotations

import gc
import os
import threading
from pathlib import Path
from typing import Any

import yaml

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows; hide the noisy warning

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/models.yaml"


def _mlx_available() -> bool:
    """True only on Apple Silicon Macs with mlx-whisper installed."""
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


class ModelRegistry:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG, tts_override: str | None = None):
        self.config_path = Path(config_path)
        self.cfg: dict[str, Any] = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) if self.config_path.exists() else {}
        self.tts_override = tts_override
        self._lock = threading.RLock()
        self._asr = None
        self._mt = None
        self._tts: dict[str, Any] = {}
        self._eval_asr = None
        self.load_counts = {"asr": 0, "mt": 0, "tts": 0, "eval_asr": 0}

    def get_asr(self):
        with self._lock:
            if self._asr is None:
                asr_cfg = self.cfg.get("asr", {})
                provider = asr_cfg.get("provider", "auto")
                use_mlx = provider == "whisper_mlx" or (provider == "auto" and _mlx_available())
                if use_mlx:
                    from ai.asr.whisper_mlx import WhisperMLX
                    self._asr = WhisperMLX(repo=asr_cfg.get("repo", "mlx-community/whisper-large-v3-turbo"))
                else:
                    # Windows, Linux, Intel Macs: faster-whisper (CPU, or NVIDIA GPU if present)
                    from ai.asr.faster_whisper_asr import FasterWhisper
                    self._asr = FasterWhisper(model=asr_cfg.get("fallback_repo", "small"),
                                              device=asr_cfg.get("fallback_device", "cpu"))
                print(f"[OMNICAST] Speech-to-text engine: {'mlx-whisper' if use_mlx else 'faster-whisper'}", flush=True)
                self.load_counts["asr"] += 1
            return self._asr

    def unload_asr(self) -> None:
        with self._lock:
            if self._asr is not None:
                self._asr = None
                gc.collect()

    def get_mt(self):
        with self._lock:
            if self._mt is None:
                mt_cfg = self.cfg.get("mt", {})
                model_id = mt_cfg.get("model_id", "facebook/nllb-200-distilled-600M")
                device = mt_cfg.get("device", "cpu")
                from ai.translation.nllb import NLLB
                self._mt = NLLB(model_id=model_id, device=device)
                self.load_counts["mt"] += 1
            return self._mt

    def get_tts(self, provider: str | None = None):
        with self._lock:
            tts_cfg = self.cfg.get("tts", {})
            chosen = provider or self.tts_override or tts_cfg.get("provider", "mms_tts")
            if chosen not in self._tts:
                if chosen == "chatterbox":
                    from ai.tts.chatterbox_client import ChatterboxTTS
                    dev = tts_cfg.get("chatterbox_device", "mps")
                    self._tts[chosen] = ChatterboxTTS(device=dev)
                elif chosen == "mms_tts":
                    from ai.tts.mms_tts import MMSTTS
                    dev = tts_cfg.get("device", "cpu")
                    self._tts[chosen] = MMSTTS(device=dev)
                else:
                    raise ValueError(f"Unknown TTS provider: {chosen}")
                self.load_counts["tts"] += 1
            return self._tts[chosen]

    def get_eval_asr(self):
        with self._lock:
            if self._eval_asr is None:
                eval_cfg = self.cfg.get("eval_asr", {})
                device = eval_cfg.get("device", "cpu")
                from ai.asr.mms_ctc import MMSCTC
                self._eval_asr = MMSCTC(device=device)
                self.load_counts["eval_asr"] += 1
            return self._eval_asr

    def close(self) -> None:
        with self._lock:
            for tts in self._tts.values():
                if hasattr(tts, "close"):
                    tts.close()
            self._tts.clear()
            self._asr = None
            self._mt = None
            self._eval_asr = None
            gc.collect()

