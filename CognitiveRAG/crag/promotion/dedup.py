from __future__ import annotations

from typing import Iterable, List

from CognitiveRAG.crag.promotion.normalizers import DurableUnit


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
        merged.provenance = {
            **dict(prev.provenance or {}),
            **dict(unit.provenance or {}),
            "dedup_count": int(dict(prev.provenance or {}).get("dedup_count", 1)) + 1,
        }
        by_key[key] = merged
    return sorted(by_key.values(), key=lambda u: (u.kind, u.normalized_key))

