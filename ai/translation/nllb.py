"""NLLB-200 MT (weights CC-BY-NC-4.0: research use only)."""
from __future__ import annotations

import time

from ai.base import ComponentResult

FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn"}


class NLLB:
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", device: str = "cpu"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
        self.device = device

    def translate(self, text: str, src: str, tgt: str, num_beams: int = 4) -> ComponentResult:
        self.tok.src_lang = FLORES[src]
        t0 = time.perf_counter()
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(FLORES[tgt]),
                                      num_beams=num_beams, max_new_tokens=256, output_scores=True,
                                      return_dict_in_generate=True)
        hyp = self.tok.batch_decode(out.sequences, skip_special_tokens=True)[0]
        dt = time.perf_counter() - t0
        score = float(out.sequences_scores[0]) if getattr(out, "sequences_scores", None) is not None else None
        return ComponentResult(hyp, self.model_id, dt, {"sequence_score": score})
