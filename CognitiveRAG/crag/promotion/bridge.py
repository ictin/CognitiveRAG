from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List

from CognitiveRAG.crag.promotion.dedup import dedup_units
from CognitiveRAG.crag.promotion.extractors import extract_prescriptions, extract_propositions
from CognitiveRAG.crag.promotion.normalizers import DurableUnit, normalize_prescription, normalize_proposition
from CognitiveRAG.schemas.memory import ReasoningPattern


def _sha16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _to_reasoning_pattern(unit: DurableUnit, *, session_id: str, source_summary: str, chunk_index: int | None) -> ReasoningPattern:
    pid = f"prom:{unit.kind}:{_sha16(unit.normalized_key)}"
    provenance_entry = {
        "session_id": session_id,
        "summary_chunk_index": chunk_index,
        "source": "context_window.v1",
        "kind": unit.kind,
        "memory_subtype": unit.memory_subtype,
        "normalized_key": unit.normalized_key,
        "source_summary": source_summary[:400],
        "source_refs": list(unit.source_refs or []),
        "source_class": "promoted_memory",
        **dict(unit.provenance or {}),
    }
    return ReasoningPattern(
        item_id=pid,
        pattern_id=pid,
        problem_signature=f"{unit.kind}:{session_id}",
        reasoning_steps=[],
        solution_summary=unit.canonical_text[:2000],
        confidence=unit.confidence,
        provenance=[json.dumps(provenance_entry)],
        memory_subtype=unit.memory_subtype,
        normalized_text=unit.normalized_key,
        freshness_state=unit.freshness_state,
        metadata={
            "durable_unit": {
                "kind": unit.kind,
                "memory_subtype": unit.memory_subtype,
                "canonical_text": unit.canonical_text,
                "normalized_key": unit.normalized_key,
                "freshness_state": unit.freshness_state,
                "source_refs": list(unit.source_refs or []),
            }
        },
        tags=[unit.kind, unit.memory_subtype],
        content=unit.canonical_text[:2000],
        summary=unit.canonical_text[:300],
    )


def promote_summaries_to_patterns(session_id: str, summaries: Iterable[Dict[str, Any]]) -> List[ReasoningPattern]:
    raw_units: list[tuple[DurableUnit, str, int | None]] = []
    for summary in summaries:
        text = str(summary.get("summary") or summary.get("text") or "").strip()
        if not text:
            continue
        chunk_index = summary.get("chunk_index")
        provenance = {
            "session_id": session_id,
            "summary_chunk_index": chunk_index,
            "created_at": summary.get("created_at"),
            "source": "session_summary",
        }

        prescriptions = extract_prescriptions(text)
        propositions = extract_propositions(text)
        for sentence in prescriptions:
            raw_units.append(
                (
                    normalize_prescription(sentence, provenance=provenance),
                    text,
                    int(chunk_index) if chunk_index is not None else None,
                )
            )
        for sentence in propositions:
            raw_units.append(
                (
                    normalize_proposition(sentence, provenance=provenance),
                    text,
                    int(chunk_index) if chunk_index is not None else None,
                )
            )

    deduped = dedup_units(unit for unit, _, __ in raw_units)
    by_key = {unit.normalized_key: unit for unit in deduped}

    patterns: List[ReasoningPattern] = []
    seen: set[str] = set()
    for unit, source_summary, chunk_index in raw_units:
        deduped_unit = by_key[unit.normalized_key]
        if deduped_unit.normalized_key in seen:
            continue
        seen.add(deduped_unit.normalized_key)
        patterns.append(
            _to_reasoning_pattern(
                deduped_unit,
                session_id=session_id,
                source_summary=source_summary,
                chunk_index=chunk_index,
            )
        )
    return patterns
