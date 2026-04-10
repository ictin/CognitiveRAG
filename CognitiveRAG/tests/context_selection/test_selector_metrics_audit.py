import json
import os
import shutil

from CognitiveRAG.session_memory.context_window import WORKDIR, assemble_context


def _seed_session(session_id: str):
    os.makedirs(WORKDIR, exist_ok=True)
    raw = [
        {"index": 0, "text": "older corpus-like copywriting note", "message_id": "m0"},
        {"index": 1, "text": "older planning summary", "message_id": "m1"},
        {"index": 2, "text": "fresh latest status message", "message_id": "m2"},
        {"index": 3, "text": "fresh followup details", "message_id": "m3"},
    ]
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f)

    summaries = [{"chunk_index": 0, "summary": "session summary about copywriting and retention"}]
    with open(os.path.join(WORKDIR, f"summaries_{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f)


def test_selector_metrics_surface_lane_and_budget_counts():
    session_id = "metrics-audit-1"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    _seed_session(session_id)

    out = assemble_context(
        session_id,
        fresh_tail_count=2,
        budget=800,
        query="investigate latest copywriting retention",
    )

    assert "selector_metrics" in out
    metrics = out["selector_metrics"]

    assert "candidate_counts" in metrics
    assert "lane_counts" in metrics
    assert "lane_tokens" in metrics
    assert "budget" in metrics
    assert "decision_stats" in metrics

    ccounts = metrics["candidate_counts"]
    assert ccounts["pre_prune"] >= ccounts["post_prune"]
    assert ccounts["pruned"] == ccounts["pre_prune"] - ccounts["post_prune"]
    assert ccounts["selected"] == len(out["selected_blocks"])
    assert ccounts["dropped"] == len(out["dropped_blocks"])

    lcounts = metrics["lane_counts"]
    assert isinstance(lcounts["pre_prune"], dict)
    assert isinstance(lcounts["selected"], dict)
    assert isinstance(lcounts["dropped"], dict)
    assert lcounts["selected"], "selected lane counts should not be empty"

    ltokens = metrics["lane_tokens"]
    assert isinstance(ltokens["pre_prune"], dict)
    assert isinstance(ltokens["selected"], dict)
    assert isinstance(ltokens["dropped"], dict)
    assert sum(ltokens["selected"].values()) == metrics["budget"]["selected_tokens"]

    budget = metrics["budget"]
    assert budget["total_budget"] == 800
    assert budget["reserved_tokens"] >= 0
    assert budget["available_budget"] == max(0, budget["total_budget"] - budget["reserved_tokens"])
    assert budget["used_total_tokens"] == budget["reserved_tokens"] + budget["selected_tokens"]
    assert 0.0 <= budget["budget_utilization_ratio"]


def test_selector_metrics_output_shape_stable_keys():
    session_id = "metrics-audit-2"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    _seed_session(session_id)

    out = assemble_context(
        session_id,
        fresh_tail_count=2,
        budget=180,
        query="what do you remember about copywriting",
    )

    metrics = out["selector_metrics"]
    assert set(metrics.keys()) == {
        "candidate_counts",
        "lane_counts",
        "lane_tokens",
        "budget",
        "decision_stats",
        "discovery",
    }

    assert set(metrics["candidate_counts"].keys()) == {"pre_prune", "post_prune", "pruned", "selected", "dropped"}
    assert set(metrics["lane_counts"].keys()) == {"pre_prune", "selected", "dropped"}
    assert set(metrics["lane_tokens"].keys()) == {"pre_prune", "selected", "dropped"}
    assert set(metrics["budget"].keys()) == {
        "total_budget",
        "reserved_tokens",
        "available_budget",
        "selected_tokens",
        "used_total_tokens",
        "budget_utilization_ratio",
    }
    assert set(metrics["decision_stats"].keys()) == {
        "drop_reasons",
        "route_intent_family",
        "route_lane_count",
        "compatibility_engine",
    }
    assert set(metrics["decision_stats"]["compatibility_engine"].keys()) == {
        "configured_mode",
        "configured_backend",
        "configured_model",
        "resolved_engine",
        "backend_available",
        "fallback_active",
        "reason",
    }
    assert set(metrics["discovery"].keys()) == {"injected_count", "injected_tokens"}
