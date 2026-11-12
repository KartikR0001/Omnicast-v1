"""Common interface for every AI component so models can be swapped without touching callers.

Every result carries the model id/version and measured wall-clock time, so that
compute cost is always a *measured* quantity (never estimated).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentResult:
    output: Any
    model_id: str
    seconds: float
    extra: dict = field(default_factory=dict)  # model-specific evidence (confidences, timings, ...)
