"""Script-safe text normalisation and error rates.

Normalisation: NFKC, casefold, drop Unicode punctuation (category P*) and symbols (S*),
KEEP combining marks (M*) so Indic vowel signs survive (cf. "What is lost in Normalization?",
EMNLP 2024). Digits are kept as written (no number verbalisation) — a documented limitation.
"""
from __future__ import annotations

import unicodedata

import jiwer


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").casefold()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return " ".join(t.split())


def wer(ref: str, hyp: str) -> float | None:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return None  # undefined without a reference; never imputed
    return jiwer.wer(r, h) if h else 1.0


def cer(ref: str, hyp: str) -> float | None:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return None
    return jiwer.cer(r, h) if h else 1.0
