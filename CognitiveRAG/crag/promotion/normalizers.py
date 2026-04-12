from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    source_refs: list[Dict[str, Any]] = field(default_factory=list)


def _freshness_from_provenance(provenance: Dict[str, Any] | None) -> str:
    payload = dict(provenance or {})
    created_at = payload.get("created_at") or payload.get("source_created_at")
    if not created_at:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0
        if age_days <= 3:
            return "hot"
        if age_days <= 30:
            return "warm"
        return "stale"
    except Exception:
        return "unknown"


def _source_ref_from_provenance(provenance: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(provenance or {})
    return {
        "session_id": payload.get("session_id"),
        "summary_chunk_index": payload.get("summary_chunk_index"),
        "created_at": payload.get("created_at"),
        "source": payload.get("source") or "session_summary",
    }


def normalize_proposition(raw: str, *, provenance: Dict[str, Any] | None = None, confidence: float = 0.7) -> DurableUnit:
    canonical = _canonicalize(raw)
    subtype = "profile_preference" if "user prefer" in canonical or "user like" in canonical else "stable_fact"
    base = max(0.0, min(1.0, float(confidence)))
    if subtype == "profile_preference":
        base = min(1.0, base + 0.06)
    return DurableUnit(
        kind="proposition",
        memory_subtype=subtype,
        canonical_text=canonical,
        normalized_key=f"proposition:{canonical}",
        confidence=base,
        freshness_state=_freshness_from_provenance(provenance),
        provenance=dict(provenance or {}),
        source_refs=[_source_ref_from_provenance(provenance)],
    )


def normalize_prescription(
    raw: str, *, provenance: Dict[str, Any] | None = None, confidence: float = 0.8
) -> DurableUnit:
    canonical = _canonicalize(raw)
    subtype = "workflow_pattern" if ("->" in canonical or "then" in canonical) else "procedure_pattern"
    base = max(0.0, min(1.0, float(confidence)))
    if subtype == "workflow_pattern":
        base = min(1.0, base + 0.05)
    return DurableUnit(
        kind="prescription",
        memory_subtype=subtype,
        canonical_text=canonical,
        normalized_key=f"prescription:{canonical}",
        confidence=base,
        freshness_state=_freshness_from_provenance(provenance),
        provenance=dict(provenance or {}),
        source_refs=[_source_ref_from_provenance(provenance)],
    )
