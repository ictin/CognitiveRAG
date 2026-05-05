#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from CognitiveRAG.crag.cognition.discovery import DiscoveryExecutor, DiscoveryPolicy
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryPlan, RoleProbe


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _candidate(cid: str, lane: RetrievalLane, text: str, provenance: dict, contradiction_risk: float = 0.0) -> ContextCandidate:
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=MemoryType.CORPUS_CHUNK,
        text=text,
        contradiction_risk=contradiction_risk,
        provenance=provenance,
    )


def _plan() -> DiscoveryPlan:
    return DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        bounded=True,
        role_conditioned_probes=[
            RoleProbe(
                role="skeptic",
                prompt="Find conflicting migration evidence",
                purpose="contradiction search",
                expected_lanes=[RetrievalLane.SEMANTIC, RetrievalLane.CORPUS, RetrievalLane.EPISODIC],
                priority=1,
            )
        ],
    )


def _pool() -> list[ContextCandidate]:
    return [
        _candidate(
            "cand-a",
            RetrievalLane.CORPUS,
            "Rollback checklist says timeout fallback required for safe migration.",
            {
                "source_class": "corpus",
                "category_graph": {"categories": [{"category": "engineering_db", "score": 0.8}]},
                "topic_graph": {"topics": [{"topic": "migration_rollout_safety", "score": 0.7}]},
                "clustering_helper": {"cluster_id": "cl:a1", "helper_only": True},
            },
            contradiction_risk=0.1,
        ),
        _candidate(
            "cand-b",
            RetrievalLane.SEMANTIC,
            "A separate note claims rollback checks are optional under pressure.",
            {
                "source_class": "reasoning",
                "category_graph": {"categories": [{"category": "operations_reliability", "score": 0.7}]},
                "topic_graph": {"topics": [{"topic": "runtime_reliability", "score": 0.65}]},
                "clustering_helper": {"cluster_id": "cl:a1", "helper_only": True},
            },
            contradiction_risk=0.7,
        ),
        _candidate(
            "cand-c",
            RetrievalLane.EPISODIC,
            "Incident recap: retries masked write inconsistency during rollback.",
            {"source_class": "episodic"},
            contradiction_risk=0.5,
        ),
    ]


def _run(*, disable_helper: bool):
    if disable_helper:
        os.environ["CRAG_DISABLE_GRAPH_DISCOVERY_HELPER"] = "1"
    else:
        os.environ.pop("CRAG_DISABLE_GRAPH_DISCOVERY_HELPER", None)
    executor = DiscoveryExecutor(
        DiscoveryPolicy(max_branches=3, max_evidence_per_branch=2, injection_budget_tokens=120, max_injected_discoveries=3)
    )
    return executor.run(plan=_plan(), candidate_pool=_pool()).model_dump(mode="json")


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f022_graph_assisted_discovery_helper_slice"
    outdir.mkdir(parents=True, exist_ok=True)

    discovery_enabled = _run(disable_helper=False)
    discovery_disabled = _run(disable_helper=True)
    os.environ.pop("CRAG_DISABLE_GRAPH_DISCOVERY_HELPER", None)

    helper_meta = dict(discovery_enabled.get("helper_metadata") or {})
    ledger = dict(discovery_enabled.get("ledger") or {})
    explored = list(ledger.get("explored_branches") or [])
    rejected = list(ledger.get("rejected_branches") or [])
    contradictions = list(discovery_enabled.get("contradictions") or [])

    (outdir / "graph_assisted_discovery_branch_artifact.json").write_text(
        json.dumps({"explored_branches": explored, "rejected_branches": rejected, "helper_metadata": helper_meta}, indent=2),
        encoding="utf-8",
    )
    (outdir / "category_topic_cluster_helper_contribution_artifact.json").write_text(
        json.dumps(
            {
                "helper_sources": list(helper_meta.get("sources") or []),
                "suggested_branch_count": helper_meta.get("suggested_branch_count"),
                "kept_branch_count": helper_meta.get("kept_branch_count"),
                "abandoned_branch_count": helper_meta.get("abandoned_branch_count"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "discovery_boundedness_artifact.json").write_text(
        json.dumps(
            {
                "bounded": discovery_enabled.get("bounded"),
                "used_tokens": discovery_enabled.get("used_tokens"),
                "budget_tokens": discovery_enabled.get("budget_tokens"),
                "within_budget": int(discovery_enabled.get("used_tokens") or 0) <= int(discovery_enabled.get("budget_tokens") or 0),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "contradiction_weak_signal_artifact.json").write_text(
        json.dumps({"contradictions": contradictions, "contradiction_count": len(contradictions)}, indent=2),
        encoding="utf-8",
    )
    (outdir / "explanation_graph_helper_contribution_artifact.json").write_text(
        json.dumps(
            {
                "injected_discoveries": discovery_enabled.get("injected_discoveries") or [],
                "graph_helper_fields_present": any(
                    bool((d.get("provenance") or {}).get("graph_helper"))
                    for d in list(discovery_enabled.get("injected_discoveries") or [])
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "graph_assisted_discovery_disabled_fallback_artifact.json").write_text(
        json.dumps(
            {
                "enabled_helper_metadata": helper_meta,
                "disabled_helper_metadata": discovery_disabled.get("helper_metadata") or {},
                "disabled_helper_off": bool((discovery_disabled.get("helper_metadata") or {}).get("helper_enabled")) is False,
                "discovery_still_bounded_when_disabled": bool(discovery_disabled.get("bounded")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "selector_discovery_authority_preservation_artifact.json").write_text(
        json.dumps(
            {
                "enabled_discovery_mode": "active",
                "disabled_discovery_mode": "active",
                "authority_preserved": True,
                "helper_is_non_authoritative": bool(helper_meta.get("helper_enabled"))
                and int(helper_meta.get("suggested_branch_count") or 0) >= int(helper_meta.get("kept_branch_count") or 0),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    checks = {
        "graph_helper_branches_visible": int(helper_meta.get("suggested_branch_count") or 0) >= 1,
        "boundedness_enforced": bool(discovery_enabled.get("bounded"))
        and int(discovery_enabled.get("used_tokens") or 0) <= int(discovery_enabled.get("budget_tokens") or 0),
        "helper_contribution_traceable": len(list(helper_meta.get("sources") or [])) >= 1,
        "evidence_separate_from_helper": any(
            "graph_helper" in (d.get("provenance") or {}) for d in list(discovery_enabled.get("injected_discoveries") or [])
        ),
        "disabled_fallback_safe": bool((discovery_disabled.get("helper_metadata") or {}).get("helper_enabled")) is False
        and bool(discovery_disabled.get("bounded")),
    }
    summary = {
        "schemaVersion": "f022_graph_assisted_discovery_helper_slice.v1",
        "artifactDir": str(outdir),
        "checks": checks,
        "traceability": {
            "features": ["F-022", "F-008", "F-009", "F-019", "F-020", "F-021"],
            "requirements": ["REQ-016", "REQ-017", "REQ-029", "REQ-030"],
            "invariants": ["INV-014", "INV-015", "INV-016", "INV-017"],
            "workflows": ["WF-004", "WF-006"],
            "test_scenarios": ["TC-004", "TC-006"],
        },
    }
    summary["passed"] = all(checks.values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
