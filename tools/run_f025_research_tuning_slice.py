#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from CognitiveRAG.crag.research_tuning.runtime_tuning_store import RuntimeTuningStore


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f025_research_tuning_slice"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    store = RuntimeTuningStore(workdir / "research_tuning.sqlite3")
    baseline_policy = store.get_effective_runtime_policy()

    eval_run = store.create_evaluation_run(
        scenario_ids=["TC-004", "TC-006", "TC-008"],
        artifact_refs=[
            "forensics/f024/summary.json",
            "forensics/f022/summary.json",
            "forensics/epic-c.json",
        ],
        metrics_before={"latency_ms": 170.0, "truthfulness_score": 0.93, "boundedness_score": 1.0},
        summary="research-tuning candidate baseline across discovery/explanation/runtime scenarios",
    )
    proposal = store.create_candidate_proposal(
        run_id=eval_run["run_id"],
        candidate_patch={
            "selector": {"bounded_discovery_branch_penalty": 0.03},
            "retrieval": {"graph_helper_bonus_cap": 0.2},
        },
        metrics_after={"latency_ms": 162.0, "truthfulness_score": 0.93, "boundedness_score": 1.0},
        rationale="bounded candidate: reduce latency with no truthfulness/boundedness regression",
    )
    after_draft_policy = store.get_effective_runtime_policy()
    rejected = store.reject_proposal(proposal["proposal_id"], reason="operator_rejected_no_runtime_deploy")
    after_reject_policy = store.get_effective_runtime_policy()

    _write(outdir / "research_evaluation_run_artifact.json", eval_run)
    _write(outdir / "candidate_tuning_proposal_artifact.json", proposal)
    _write(outdir / "before_after_comparison_artifact.json", proposal["comparison"])
    _write(
        outdir / "non_authoritative_proposal_artifact.json",
        {
            "proposal_id": proposal["proposal_id"],
            "status": proposal["status"],
            "authoritative": proposal["authoritative"],
            "requires_explicit_acceptance": proposal["requires_explicit_acceptance"],
            "runtime_policy_unchanged": baseline_policy == after_draft_policy,
        },
    )
    _write(outdir / "rejection_rollback_safety_artifact.json", rejected["rollback_state"])
    _write(
        outdir / "explanation_tuning_contribution_artifact.json",
        {
            "proposal_id": proposal["proposal_id"],
            "rationale": proposal["rationale"],
            "explanation": "Suggestion derives from evaluation metrics; proposal remains non-authoritative until explicit acceptance.",
        },
    )

    checks = {
        "evaluation_run_recorded": bool(eval_run.get("run_id")),
        "proposal_non_authoritative_by_default": proposal.get("status") == RuntimeTuningStore.STATUS_DRAFT
        and proposal.get("authoritative") is False,
        "before_after_comparison_present": bool((proposal.get("comparison") or {}).get("delta")),
        "no_runtime_change_without_approval": baseline_policy == after_draft_policy,
        "rollback_safety_recorded": bool(rejected.get("rollback_state", {}).get("rollback_ready")),
        "rejected_proposal_non_authoritative": rejected.get("status") == RuntimeTuningStore.STATUS_REJECTED and rejected.get("authoritative") is False,
        "runtime_policy_still_unchanged_after_reject": baseline_policy == after_reject_policy,
    }
    summary = {
        "schemaVersion": "f025_research_tuning_slice.v1",
        "artifactDir": str(outdir),
        "checks": checks,
        "traceability": {
            "features": ["F-025", "F-017", "F-018", "F-022", "F-024", "F-009"],
            "requirements": ["REQ-016", "REQ-017", "REQ-018", "REQ-031", "REQ-032", "REQ-034"],
            "invariants": ["INV-004", "INV-009", "INV-016", "INV-017", "INV-024"],
            "workflows": ["WF-004", "WF-006", "WF-007"],
            "test_scenarios": ["TC-004", "TC-006", "TC-008"],
        },
    }
    summary["passed"] = all(checks.values())
    _write(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
