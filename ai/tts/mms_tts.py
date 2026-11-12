"""Meta MMS-TTS (VITS architecture) wrapper (`facebook/mms-tts-*`).

Provides fast, local multilingual synthesis (`hi`, `fr`, `de`, `es`, `en`) behind the same
ComponentResult contract as ChatterboxTTS. Seeded stochastic sampling (`torch.manual_seed`)
ensures REGENERATE_TTS produces distinct acoustic realizations across attempts.
Weights licence: CC-BY-NC-4.0 (research / academic review use).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ai.base import ComponentResult

MMS_TTS_REPOS = {
    "hi": "facebook/mms-tts-hin",
    "fr": "facebook/mms-tts-fra",
    "de": "facebook/mms-tts-deu",
    "es": "facebook/mms-tts-spa",
    "en": "facebook/mms-tts-eng",
}


class MMSTTS:
    """Lazy per-language VITS synthesizer implementing the OMNICAST TTSModel interface."""

    def __init__(self, device: str = "cpu"):
        import torch
        self.torch = torch
        self.device = device
        self._models: dict[str, tuple] = {}
        self.load_seconds: float = 0.0

    def _get_lang_model(self, lang: str):
        if lang not in MMS_TTS_REPOS:
            raise ValueError(f"Unsupported MMS-TTS language '{lang}'. Supported: {sorted(MMS_TTS_REPOS)}")
        if lang not in self._models:
            from transformers import AutoTokenizer, VitsModel
            repo = MMS_TTS_REPOS[lang]
            t0 = time.perf_counter()
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = VitsModel.from_pretrained(repo).to(self.device).eval()
            self.load_seconds += time.perf_counter() - t0
            self._models[lang] = (repo, tok, mdl)
        return self._models[lang]

    def synthesize(
        self,
        text: str,
        language: str,
        prompt_wav: str | None = None,
        out_wav: str = "out.wav",
        seed: int = 0,
        speaking_rate: float = 1.0,
        noise_scale: float = 0.667,
    ) -> ComponentResult:
        repo, tok, mdl = self._get_lang_model(language)
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text")

        self.torch.manual_seed(int(seed))
        mdl.speaking_rate = float(speaking_rate)
        mdl.noise_scale = float(noise_scale)
        mdl.noise_scale_duration = 0.8

        t0 = time.perf_counter()
        inputs = tok(clean_text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[-1] == 0:
            raise ValueError(f"Tokenizer produced 0 tokens for text: {clean_text!r}")
        with self.torch.inference_mode():
            wav = mdl(**inputs).waveform[0].detach().cpu().numpy().astype(np.float32)
        dt = time.perf_counter() - t0

        sr = int(mdl.config.sampling_rate)
        out_path = Path(out_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, wav, sr)
        audio_s = len(wav) / sr if sr else 0.0
        return ComponentResult(
            str(out_path),
            repo,
            dt,
            {
                "ok": True,
                "out_wav": str(out_path),
                "seconds": dt,
                "audio_seconds": audio_s,
                "sample_rate": sr,
                "seed": int(seed),
                "prompt_used": False,
            },
        )

    def close(self) -> None:
        self._models.clear()

