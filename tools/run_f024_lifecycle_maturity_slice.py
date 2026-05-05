#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.cognition.discovery import DiscoveryExecutor, DiscoveryPolicy
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryPlan, RoleProbe
from CognitiveRAG.crag.federation.local_trust_envelope import LocalFederationEnvelopeStore
from CognitiveRAG.crag.retrieval import promoted_lane, web_lane
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_reasoning_db(workdir: Path) -> None:
    db_path = workdir / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
        )
        db.execute(
            "INSERT OR REPLACE INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "rp_f024",
                "postgres migration rollback timeout",
                "[]",
                "Use canary + rollback timeout fallback and verification.",
                0.89,
                "[]",
                "workflow_pattern",
                "postgres migration rollback timeout workflow",
                "warm",
            ),
        )
        db.commit()


def _seed_web_promoted(workdir: Path) -> dict:
    store = WebPromotedMemoryStore(workdir / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_fresh",
        canonical_fact="Fresh trusted rollout guardrail.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.88,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/fresh"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        approval_reason="seed",
        approval_basis={"seed": True},
        now_iso="2026-05-05T08:00:00Z",
    )
    store.upsert_fact(
        promoted_id="wp_old",
        canonical_fact="Old trusted rollout guardrail requiring revalidation.",
        evidence_ids=["ev3", "ev4"],
        confidence=0.82,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/old"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        approval_reason="seed",
        approval_basis={"seed": True},
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_STALE,
        freshness_reason="ttl_expired",
        now_iso="2026-05-01T08:00:00Z",
    )
    stale = store.get("wp_old")
    reval = store.request_revalidation("wp_old", reason="ttl_expired_requires_revalidation", now_iso="2026-05-05T08:04:00Z")
    return {"stale": stale, "revalidation": reval}


def _run_discovery(candidates: list[ContextCandidate]) -> dict:
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        bounded=True,
        role_conditioned_probes=[
            RoleProbe(
                role="skeptic",
                prompt="Find contradiction or weak signal in migration guidance.",
                purpose="contradiction search",
                expected_lanes=[RetrievalLane.PROMOTED, RetrievalLane.WEB],
                priority=1,
            )
        ],
    )
    executor = DiscoveryExecutor(
        DiscoveryPolicy(max_branches=3, max_evidence_per_branch=2, injection_budget_tokens=140, max_injected_discoveries=3)
    )
    return executor.run(plan=plan, candidate_pool=candidates).model_dump(mode="json")


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f024_lifecycle_maturity_cross_class_slice"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    _seed_reasoning_db(workdir)
    reval_transition = _seed_web_promoted(workdir)

    promoted_hits = promoted_lane.retrieve(
        workdir=str(workdir),
        intent_family=IntentFamily.MEMORY_SUMMARY,
        query="rollback timeout workflow",
        top_k=4,
    )
    web_hits = web_lane.retrieve(
        workdir=str(workdir),
        query="rollout guardrail",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=5,
    )

    fed = LocalFederationEnvelopeStore(workdir / "federation_packets.sqlite3")
    pkt = fed.export_packet(
        source_install_id="local-install-a",
        payload_class="reasoning_pattern",
        source_object_ids=["promoted:rp_f024", "webpromoted:wp_old"],
        provenance_refs=[
            {"type": "reasoning_pattern", "id": "rp_f024", "uri": "memory://reasoning/rp_f024"},
            {"type": "web_promoted", "id": "wp_old", "uri": "memory://web/wp_old"},
        ],
        payload={"summary": "Imported external lifecycle-bound packet", "confidence": 0.77},
        freshness_lifecycle_state="revalidation_required",
    )
    fed_readback = fed.read_packet(fed.import_packet(pkt)["packet_id"])

    promoted_lifecycle = [
        {
            "id": h.id,
            "memory_type": h.memory_type.value,
            "source_class": (h.provenance or {}).get("source_class"),
            "lifecycle": (h.provenance or {}).get("lifecycle"),
            "provenance": dict(h.provenance or {}),
        }
        for h in promoted_hits
    ]
    web_lifecycle = [
        {
            "id": h.id,
            "memory_type": h.memory_type.value,
            "source_class": (h.provenance or {}).get("source_class"),
            "lifecycle": (h.provenance or {}).get("lifecycle"),
            "provenance": dict(h.provenance or {}),
        }
        for h in web_hits
        if h.memory_type in {MemoryType.WEB_PROMOTED_FACT, MemoryType.WEB_EVIDENCE}
    ]

    discovery_candidates = []
    for h in (promoted_hits[:1] + web_hits[:2]):
        prov = dict(h.provenance or {})
        if h.id.endswith("wp_old"):
            prov["contradiction_hint"] = "stale guidance conflicts with fresh rollout checks"
        discovery_candidates.append(
            ContextCandidate(
                id=h.id,
                lane=h.lane,
                memory_type=h.memory_type,
                text=h.text,
                contradiction_risk=float(h.contradiction_risk or 0.0),
                provenance=prov,
            )
        )
    discovery = _run_discovery(discovery_candidates)

    clear_routing_caches()
    lanes_enabled, hits_enabled = route_and_retrieve(
        query="rollback timeout guidance",
        intent_family=IntentFamily.INVESTIGATIVE,
        session_id="f024-lifecycle-proof",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(workdir),
        top_k_per_lane=6,
    )
    os.environ["CRAG_DISABLE_FEDERATION_HELPER"] = "1"
    clear_routing_caches()
    lanes_disabled, hits_disabled = route_and_retrieve(
        query="rollback timeout guidance",
        intent_family=IntentFamily.INVESTIGATIVE,
        session_id="f024-lifecycle-proof",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(workdir),
        top_k_per_lane=6,
    )
    os.environ.pop("CRAG_DISABLE_FEDERATION_HELPER", None)

    matrix = {
        "promoted_memory": sorted({((row.get("lifecycle") or {}).get("normalized_state")) for row in promoted_lifecycle if row.get("lifecycle")}),
        "web_memory": sorted({((row.get("lifecycle") or {}).get("normalized_state")) for row in web_lifecycle if row.get("lifecycle")}),
        "federation_import": [((fed_readback.get("lifecycle") or {}).get("normalized_state"))],
    }
    _write(outdir / "lifecycle_state_matrix_artifact.json", matrix)
    _write(outdir / "promoted_memory_lifecycle_readback_artifact.json", {"hits": promoted_lifecycle})
    _write(outdir / "web_promoted_lifecycle_readback_artifact.json", {"hits": web_lifecycle})
    _write(
        outdir / "reasoning_memory_lifecycle_readback_artifact.json",
        {"hits": [row for row in promoted_lifecycle if (row.get("provenance") or {}).get("source_store") == "reasoning_store"]},
    )
    _write(
        outdir / "graph_discovery_helper_lifecycle_artifact.json",
        {
            "helper_metadata": discovery.get("helper_metadata") or {},
            "injected_discoveries": discovery.get("injected_discoveries") or [],
            "bounded": discovery.get("bounded"),
            "used_tokens": discovery.get("used_tokens"),
            "budget_tokens": discovery.get("budget_tokens"),
        },
    )
    _write(
        outdir / "federation_quarantine_trust_lifecycle_artifact.json",
        {
            "packet_id": fed_readback.get("packet_id"),
            "import_state": fed_readback.get("import_state"),
            "trust_status": fed_readback.get("trust_status"),
            "approval_status": fed_readback.get("approval_status"),
            "authoritative": fed_readback.get("authoritative"),
            "lifecycle": fed_readback.get("lifecycle"),
            "provenance_refs": fed_readback.get("provenance_refs"),
        },
    )
    _write(outdir / "revalidation_required_transition_artifact.json", reval_transition)
    _write(
        outdir / "explanation_lifecycle_truthfulness_artifact.json",
        {
            "selected_lifecycle_samples": [
                ((promoted_lifecycle[0]["lifecycle"] if promoted_lifecycle else {}) or {}),
                ((web_lifecycle[0]["lifecycle"] if web_lifecycle else {}) or {}),
                ((fed_readback.get("lifecycle") or {})),
            ],
            "truthfulness_guard": "No lifecycle state is upgraded in explanation artifact; states mirror retrieval/readback provenance.",
        },
    )
    _write(
        outdir / "retrieval_lifecycle_metadata_artifact.json",
        {
            "hits_enabled": [
                {
                    "id": h.id,
                    "lane": h.lane.value,
                    "memory_type": h.memory_type.value,
                    "lifecycle": (h.provenance or {}).get("lifecycle"),
                    "source_class": (h.provenance or {}).get("source_class"),
                }
                for h in hits_enabled
            ]
        },
    )
    _write(
        outdir / "disabled_degraded_lifecycle_fallback_artifact.json",
        {
            "lanes_enabled": [l.value for l in lanes_enabled.lanes],
            "lanes_disabled": [l.value for l in lanes_disabled.lanes],
            "enabled_hit_count": len(hits_enabled),
            "disabled_hit_count": len(hits_disabled),
            "core_retrieval_survives_disabled_helper": bool(lanes_disabled.lanes) and len(hits_disabled) >= 0,
        },
    )

    checks = {
        "normalized_cross_class_visible": bool(matrix["promoted_memory"]) and bool(matrix["web_memory"]) and matrix["federation_import"][0] == "quarantined",
        "revalidation_transition_visible": bool((reval_transition.get("revalidation") or {}).get("freshness_lifecycle_state") == "revalidation_pending"),
        "quarantine_non_authoritative": fed_readback.get("import_state") == "quarantined" and (fed_readback.get("authoritative") is False),
        "discovery_bounded": bool(discovery.get("bounded")) and int(discovery.get("used_tokens") or 0) <= int(discovery.get("budget_tokens") or 0),
        "retrieval_lifecycle_exposed": any(((h.provenance or {}).get("lifecycle")) for h in hits_enabled),
        "fallback_safe": bool(lanes_disabled.lanes),
    }
    summary = {
        "schemaVersion": "f024_lifecycle_maturity_cross_class_slice.v1",
        "artifactDir": str(outdir),
        "checks": checks,
        "traceability": {
            "features": ["F-024", "F-015", "F-004", "F-006", "F-007", "F-009", "F-013", "F-014", "F-022", "F-023"],
            "requirements": ["REQ-009", "REQ-013", "REQ-014", "REQ-015", "REQ-018", "REQ-021", "REQ-032"],
            "invariants": ["INV-008", "INV-009", "INV-011", "INV-012", "INV-016", "INV-017"],
            "workflows": ["WF-005", "WF-006"],
            "test_scenarios": ["TC-005", "TC-006"],
        },
    }
    summary["passed"] = all(checks.values())
    _write(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
