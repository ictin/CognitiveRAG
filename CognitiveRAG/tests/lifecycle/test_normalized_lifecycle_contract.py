from __future__ import annotations

from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType
from CognitiveRAG.crag.federation.local_trust_envelope import LocalFederationEnvelopeStore
from CognitiveRAG.crag.lifecycle.normalization import normalized_lifecycle_view
from CognitiveRAG.crag.retrieval import promoted_lane, web_lane
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def test_normalized_contract_maps_core_states():
    assert normalized_lifecycle_view(source_class="promoted_memory", provenance={"promotion_state": "trusted", "approval_status": "approved", "freshness_lifecycle_state": "fresh"})["normalized_state"] == "trusted"
    assert normalized_lifecycle_view(source_class="web_promoted", provenance={"promotion_state": "trusted", "freshness_lifecycle_state": "stale"})["normalized_state"] == "stale"
    assert normalized_lifecycle_view(source_class="web_promoted", provenance={"promotion_state": "trusted", "freshness_lifecycle_state": "revalidation_pending"})["normalized_state"] == "revalidation_required"
    assert normalized_lifecycle_view(source_class="federation_import", provenance={"import_state": "quarantined", "authoritative": False})["normalized_state"] == "quarantined"
    assert normalized_lifecycle_view(source_class="promoted_memory", provenance={"promotion_state": "trusted", "approval_status": "approved", "freshness_lifecycle_state": "revalidated"})["normalized_state"] == "revalidated"


def test_retrieval_surfaces_normalized_lifecycle_for_promoted_reasoning(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
        )
        db.execute(
            "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "rp_lifecycle",
                "Use canary rollout + timeout fallback.",
                0.88,
                "[]",
                "workflow_pattern",
                "canary rollout timeout fallback",
                "warm",
            ),
        )
        db.commit()

    hits = promoted_lane.retrieve(
        workdir=str(tmp_path),
        intent_family=IntentFamily.MEMORY_SUMMARY,
        query="canary rollout timeout",
        top_k=4,
    )
    assert hits
    lifecycle = (hits[0].provenance or {}).get("lifecycle") or {}
    assert lifecycle.get("normalized_state") == "trusted"


def test_web_and_federation_surfaces_expose_non_authoritative_lifecycle(tmp_path: Path, monkeypatch):
    web_store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    web_store.upsert_fact(
        promoted_id="wp_old",
        canonical_fact="Legacy migration pattern.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.81,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/old"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        approval_reason="seed",
        approval_basis={"seed": True},
        now_iso="2026-04-03T10:00:00Z",
    )
    web_store.request_revalidation("wp_old", reason="ttl_expired", now_iso="2026-04-03T10:05:00Z")

    monkeypatch.setattr(web_lane.WebFetcher, "fetch_plan", lambda self, plan, need, min_cache_hits=2: [])
    hits = web_lane.retrieve(workdir=str(tmp_path), query="Legacy migration pattern", intent_family=IntentFamily.INVESTIGATIVE, top_k=4)
    promoted = [h for h in hits if h.memory_type == MemoryType.WEB_PROMOTED_FACT]
    assert promoted
    lifecycle = (promoted[0].provenance or {}).get("lifecycle") or {}
    assert lifecycle.get("normalized_state") in {"revalidation_required", "stale", "trusted"}

    fed = LocalFederationEnvelopeStore(tmp_path / "federation_packets.sqlite3")
    pkt = fed.export_packet(
        source_install_id="local-install-a",
        payload_class="reasoning_pattern",
        source_object_ids=["promoted:rp_lifecycle"],
        provenance_refs=[{"type": "reasoning_pattern", "id": "rp_lifecycle", "uri": "memory://reasoning/rp_lifecycle"}],
        payload={"summary": "Imported reusable reasoning"},
        freshness_lifecycle_state="revalidation_required",
    )
    imported = fed.import_packet(pkt)
    readback = fed.read_packet(imported["packet_id"])
    fed_lifecycle = readback.get("lifecycle") or {}
    assert fed_lifecycle.get("normalized_state") == "quarantined"
    assert readback.get("authoritative") is False
