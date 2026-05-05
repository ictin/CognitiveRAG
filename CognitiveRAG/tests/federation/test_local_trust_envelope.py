from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.federation.local_trust_envelope import LocalFederationEnvelopeStore, stable_packet_id
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve


def _sample_packet_inputs():
    return {
        "source_install_id": "local-install-a",
        "payload_class": "reasoning_pattern",
        "source_object_ids": ["promoted:abc", "webpromoted:xyz"],
        "provenance_refs": [
            {"type": "promoted_memory", "id": "abc", "uri": "memory://promoted/abc"},
            {"type": "web_promoted", "id": "xyz", "uri": "memory://web/xyz"},
        ],
        "payload": {"summary": "Rollback strategy with guardrails", "confidence": 0.81},
    }


def test_packet_id_is_stable_for_same_inputs():
    args = _sample_packet_inputs()
    a = stable_packet_id(**args)
    b = stable_packet_id(**args)
    assert a == b
    assert a.startswith("fedpkt:")


def test_import_is_quarantined_non_authoritative_and_untrusted(tmp_path: Path):
    store = LocalFederationEnvelopeStore(tmp_path / "federation_packets.sqlite3")
    packet = store.export_packet(**_sample_packet_inputs(), freshness_lifecycle_state="revalidation_required")
    imported = store.import_packet(packet)

    assert imported["packet_id"] == packet["packet_id"]
    assert imported["import_state"] == LocalFederationEnvelopeStore.IMPORT_STATE_QUARANTINED
    assert imported["trust_status"] == LocalFederationEnvelopeStore.TRUST_UNTRUSTED
    assert imported["approval_status"] == LocalFederationEnvelopeStore.APPROVAL_UNREVIEWED
    assert imported["authoritative"] is False
    assert imported["freshness_lifecycle_state"] == "revalidation_required"
    assert len(imported["provenance_refs"]) >= 1


def test_federation_store_does_not_change_core_retrieval_path(tmp_path: Path):
    # Seed one local federation packet; retrieval should remain core-lane driven.
    store = LocalFederationEnvelopeStore(tmp_path / "federation_packets.sqlite3")
    packet = store.export_packet(**_sample_packet_inputs())
    store.import_packet(packet)

    clear_routing_caches()
    plan, hits = route_and_retrieve(
        query="postgres migration rollback timeout",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="federation-fallback-proof",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )
    assert plan.lanes  # core retrieval still executes
    assert all("federation" not in dict(h.provenance or {}) for h in hits)
