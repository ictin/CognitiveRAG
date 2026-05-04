from __future__ import annotations

import os
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve


def _run(query: str, *, workdir: str, legacy: bool):
    if legacy:
        os.environ["CRAG_F017_LEGACY_SORT"] = "1"
    else:
        os.environ.pop("CRAG_F017_LEGACY_SORT", None)
    clear_routing_caches()
    plan, hits = route_and_retrieve(
        query=query,
        intent_family=IntentFamily.INVESTIGATIVE,
        session_id="f017-eq",
        fresh_tail=[{"index": 0, "text": "f017 seed fresh tail", "sender": "user"}],
        older_raw=[{"index": 1, "text": "f017 older raw evidence", "sender": "assistant"}],
        summaries=[{"chunk_index": 0, "summary": "f017 summary seed"}],
        workdir=workdir,
        top_k_per_lane=8,
    )
    return plan, hits


def test_f017_hot_path_sort_equivalence_and_metadata_preservation(tmp_path: Path):
    q = "f017 deterministic route cache controls"
    plan_legacy, hits_legacy = _run(q, workdir=str(tmp_path), legacy=True)
    plan_opt, hits_opt = _run(q, workdir=str(tmp_path), legacy=False)

    assert [x.value for x in plan_legacy.lanes] == [x.value for x in plan_opt.lanes]
    assert [h.id for h in hits_legacy] == [h.id for h in hits_opt]

    for a, b in zip(hits_legacy, hits_opt):
        assert a.lane == b.lane
        assert a.memory_type == b.memory_type
        assert (a.provenance or {}).get("source_class") == (b.provenance or {}).get("source_class")
        assert (a.provenance or {}).get("lifecycle") == (b.provenance or {}).get("lifecycle")
