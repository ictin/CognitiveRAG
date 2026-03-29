from __future__ import annotations

from typing import Any, Dict


Provenance = Dict[str, Any]
WeightsMap = Dict[str, float]


def estimate_tokens(text: str) -> int:
    """Stable token estimate used by selector tests and candidate normalization."""
    if not text:
        return 0
    # Deterministic rough estimate (chars/4) with floor of 1 for non-empty text.
    return max(1, (len(text) + 3) // 4)


def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value
