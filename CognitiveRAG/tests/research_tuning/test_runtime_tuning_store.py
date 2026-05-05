from __future__ import annotations

from pathlib import Path

from CognitiveRAG.crag.research_tuning.runtime_tuning_store import RuntimeTuningStore


def test_evaluation_run_record_is_stable_and_readable(tmp_path: Path):
    store = RuntimeTuningStore(tmp_path / "research_tuning.sqlite3")
    run = store.create_evaluation_run(
        scenario_ids=["TC-004", "TC-006", "TC-008"],
        artifact_refs=["forensics/a.json", "forensics/b.json"],
        metrics_before={"latency_ms": 180.0, "truthfulness_score": 0.92},
        summary="baseline evaluation before candidate tuning",
    )
    loaded = store.get_evaluation_run(run["run_id"])
    assert loaded is not None
    assert loaded["run_id"] == run["run_id"]
    assert loaded["scenario_ids"] == sorted(["TC-004", "TC-006", "TC-008"])


def test_candidate_proposal_defaults_non_authoritative_and_requires_acceptance(tmp_path: Path):
    store = RuntimeTuningStore(tmp_path / "research_tuning.sqlite3")
    run = store.create_evaluation_run(
        scenario_ids=["TC-004"],
        artifact_refs=["forensics/base.json"],
        metrics_before={"latency_ms": 200.0, "truthfulness_score": 0.9},
        summary="baseline",
    )
    proposal = store.create_candidate_proposal(
        run_id=run["run_id"],
        candidate_patch={"selector": {"novelty_weight": 0.52}},
        metrics_after={"latency_ms": 185.0, "truthfulness_score": 0.9},
        rationale="reduce latency while preserving truthfulness",
    )
    assert proposal["status"] == RuntimeTuningStore.STATUS_DRAFT
    assert proposal["authoritative"] is False
    assert proposal["requires_explicit_acceptance"] is True
    assert proposal["comparison"]["delta"]["latency_ms"] == -15.0


def test_runtime_policy_unchanged_without_explicit_acceptance_and_reject_has_rollback(tmp_path: Path):
    store = RuntimeTuningStore(tmp_path / "research_tuning.sqlite3")
    baseline = store.get_effective_runtime_policy()
    run = store.create_evaluation_run(
        scenario_ids=["TC-006"],
        artifact_refs=["forensics/explain.json"],
        metrics_before={"explain_fidelity": 0.93},
        summary="baseline explanation fidelity",
    )
    proposal = store.create_candidate_proposal(
        run_id=run["run_id"],
        candidate_patch={"explanation": {"lane_weight_delta": 0.01}},
        metrics_after={"explain_fidelity": 0.94},
        rationale="small bounded lane weighting suggestion",
    )
    after_draft = store.get_effective_runtime_policy()
    assert after_draft == baseline

    rejected = store.reject_proposal(proposal["proposal_id"], reason="operator rejected proposal")
    assert rejected["status"] == RuntimeTuningStore.STATUS_REJECTED
    assert rejected["rollback_state"]["rollback_ready"] is True
    assert rejected["rollback_state"]["runtime_changed"] is False
    assert store.get_effective_runtime_policy() == baseline
