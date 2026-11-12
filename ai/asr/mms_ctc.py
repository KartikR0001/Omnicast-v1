"""Meta MMS-1b-all CTC ASR (non-Whisper family). Used as the HELD-OUT evaluator ASR.

Weights licence: CC-BY-NC-4.0 (research use only).
"""
from __future__ import annotations

import time

import numpy as np

from ai.base import ComponentResult

ISO3 = {"en": "eng", "hi": "hin", "fr": "fra", "de": "deu", "es": "spa"}
MODEL_ID = "facebook/mms-1b-all"


class MMSCTC:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()
        self.device = device
        self.lang = None

    def _set_lang(self, lang: str):
        code = ISO3[lang]
        if code != self.lang:
            self.processor.tokenizer.set_target_lang(code)
            self.model.load_adapter(code)
            self.lang = code

    def transcribe(self, audio_16k: np.ndarray, language: str) -> ComponentResult:
        self._set_lang(language)
        t0 = time.perf_counter()
        inputs = self.processor(audio_16k.astype(np.float32), sampling_rate=16000, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        ids = logits.argmax(-1)[0]
        text = self.processor.decode(ids)
        probs = logits.softmax(-1)[0].max(-1).values
        dt = time.perf_counter() - t0
        return ComponentResult(text.strip(), f"{MODEL_ID}:{self.lang}", dt, {"mean_frame_maxprob": float(probs.mean())})
