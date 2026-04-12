from __future__ import annotations

from typing import Iterable, List

from CognitiveRAG.crag.promotion.normalizers import DurableUnit


def _merge_source_refs(*refs_lists: list[dict] | None) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for refs in refs_lists:
        for ref in list(refs or []):
            key = "|".join(
                [
                    str(ref.get("session_id") or ""),
                    str(ref.get("summary_chunk_index") or ""),
                    str(ref.get("created_at") or ""),
                    str(ref.get("source") or ""),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(ref))
    return merged


def dedup_units(units: Iterable[DurableUnit]) -> List[DurableUnit]:
    by_key: dict[str, DurableUnit] = {}
    for unit in units:
        key = unit.normalized_key
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = unit
            continue
        if unit.confidence > prev.confidence:
            merged = unit
        else:
            merged = prev
        merged.source_refs = _merge_source_refs(prev.source_refs, unit.source_refs)
        merged.provenance = {
            **dict(prev.provenance or {}),
            **dict(unit.provenance or {}),
            "dedup_count": int(dict(prev.provenance or {}).get("dedup_count", 1)) + 1,
            "source_refs": merged.source_refs,
        }
        by_key[key] = merged
    return sorted(by_key.values(), key=lambda u: (u.kind, u.normalized_key))
