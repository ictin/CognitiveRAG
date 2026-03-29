from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict

from CognitiveRAG.crag.promotion.speech_act import strip_speech_act


def _canonicalize(text: str) -> str:
    value = strip_speech_act(text).lower()
    value = value.replace("→", "->")
    value = re.sub(r"\b(the user|user)\s+said\s+", "user ", value)
    value = re.sub(r"\bpreferences?\b", "prefers", value)
    value = re.sub(r"[^a-z0-9\s\-\>:/]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


@dataclass(slots=True)
class DurableUnit:
    kind: str
    memory_subtype: str
    canonical_text: str
    normalized_key: str
    confidence: float
    freshness_state: str
    provenance: Dict[str, Any] = field(default_factory=dict)


def normalize_proposition(raw: str, *, provenance: Dict[str, Any] | None = None, confidence: float = 0.7) -> DurableUnit:
    canonical = _canonicalize(raw)
    subtype = "profile_preference" if "user prefer" in canonical or "user like" in canonical else "stable_fact"
    return DurableUnit(
        kind="proposition",
        memory_subtype=subtype,
        canonical_text=canonical,
        normalized_key=f"proposition:{canonical}",
        confidence=max(0.0, min(1.0, confidence)),
        freshness_state="current",
        provenance=dict(provenance or {}),
    )


def normalize_prescription(
    raw: str, *, provenance: Dict[str, Any] | None = None, confidence: float = 0.8
) -> DurableUnit:
    canonical = _canonicalize(raw)
    subtype = "workflow_pattern" if ("->" in canonical or "then" in canonical) else "procedure_pattern"
    return DurableUnit(
        kind="prescription",
        memory_subtype=subtype,
        canonical_text=canonical,
        normalized_key=f"prescription:{canonical}",
        confidence=max(0.0, min(1.0, confidence)),
        freshness_state="current",
        provenance=dict(provenance or {}),
    )

