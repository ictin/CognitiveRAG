import json
import os
import shutil

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryResult
from CognitiveRAG.crag.retrieval.router import RoutePlan
from CognitiveRAG.session_memory.context_window import WORKDIR, assemble_context, compact_session


def test_assemble_context_uses_selector_and_emits_explanation(tmp_path):
    session_id = "selector-integ"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)

    raw_path = os.path.join(WORKDIR, f"raw_{session_id}.json")
    raw = [{"index": i, "text": f"message {i}", "message_id": f"m{i}"} for i in range(25)]
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    compact_session(session_id, older_than_index=15)
    out = assemble_context(session_id, fresh_tail_count=4, budget=600, query="what do you remember?")

    assert "fresh_tail" in out and "summaries" in out
    assert "explanation" in out and isinstance(out["explanation"], dict)
    assert out["explanation"]["intent_family"] == "memory_summary"
    assert len(out["fresh_tail"]) >= 1
    assert "selected_blocks" in out


def test_assemble_context_enforces_reservation_and_global_budget_tradeoffs(monkeypatch, tmp_path):
    session_id = "selector-budget-integ"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)

    raw_path = os.path.join(WORKDIR, f"raw_{session_id}.json")
    raw = [{"index": i, "text": ("fresh reserve text " * 6) if i >= 10 else f"message {i}", "message_id": f"m{i}"} for i in range(12)]
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    route_plan = RoutePlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        lanes=[RetrievalLane.FRESH_TAIL, RetrievalLane.SEMANTIC, RetrievalLane.CORPUS],
        reason="test-route",
        metadata={},
    )
    lane_candidates = [
        ContextCandidate(
            id="f1",
            lane=RetrievalLane.FRESH_TAIL,
            memory_type=MemoryType.EPISODIC_RAW,
            text="fresh tail one",
            tokens=20,
            provenance={"message": {"text": "fresh tail one"}},
            lexical_score=0.05,
            semantic_score=0.05,
        ),
        ContextCandidate(
            id="f2",
            lane=RetrievalLane.FRESH_TAIL,
            memory_type=MemoryType.EPISODIC_RAW,
            text="fresh tail two",
            tokens=20,
            provenance={"message": {"text": "fresh tail two"}},
            lexical_score=0.05,
            semantic_score=0.05,
        ),
        ContextCandidate(
            id="s1",
            lane=RetrievalLane.SEMANTIC,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="high utility semantic 1",
            tokens=80,
            provenance={"source": "semantic"},
            lexical_score=0.9,
            semantic_score=0.95,
            novelty_score=0.8,
        ),
        ContextCandidate(
            id="s1dup",
            lane=RetrievalLane.SEMANTIC,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="high utility semantic 1",
            tokens=80,
            provenance={"source": "semantic-dup"},
            lexical_score=0.88,
            semantic_score=0.93,
            novelty_score=0.8,
        ),
        ContextCandidate(
            id="s2",
            lane=RetrievalLane.SEMANTIC,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="high utility semantic 2",
            tokens=80,
            provenance={"source": "semantic"},
            lexical_score=0.88,
            semantic_score=0.92,
            novelty_score=0.75,
        ),
    ]

    monkeypatch.setattr(
        "CognitiveRAG.session_memory.context_window.build_candidates_with_route",
        lambda **_: (route_plan, lane_candidates),
    )
    monkeypatch.setattr(
        "CognitiveRAG.session_memory.context_window.DiscoveryExecutor.run",
        lambda self, **_: DiscoveryResult(budget_tokens=220, used_tokens=0, injected_discoveries=[]),
    )

    out = assemble_context(
        session_id,
        fresh_tail_count=2,
        budget=420,
        query="investigate lane tradeoffs",
        intent_family=IntentFamily.INVESTIGATIVE,
    )

    metrics = out["selector_metrics"]
    budget_metrics = metrics["budget"]
    selected_ids = {block["id"] for block in out["selected_blocks"]}

    assert budget_metrics["reserved_tokens"] > 256
    assert budget_metrics["used_total_tokens"] <= budget_metrics["total_budget"]
    assert metrics["candidate_counts"]["pruned"] >= 1

    # Enforced lane minima plus reservation budget means semantic high-utility items
    # can be dropped when they do not fit globally under budget.
    assert {"f1", "f2"} <= selected_ids
    assert "s1" not in selected_ids and "s2" not in selected_ids


def test_assemble_context_runtime_nli_path_changes_selection_and_reports_engine(monkeypatch, tmp_path):
    session_id = "selector-nli-runtime-integ"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)

    raw_path = os.path.join(WORKDIR, f"raw_{session_id}.json")
    raw = [{"index": i, "text": f"message {i}", "message_id": f"m{i}"} for i in range(8)]
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    route_plan = RoutePlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        lanes=[RetrievalLane.CORPUS],
        reason="test-route",
        metadata={},
    )
    lane_candidates = [
        ContextCandidate(
            id="a",
            lane=RetrievalLane.CORPUS,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="alpha release checklist approved",
            tokens=20,
            provenance={"source": "corpus"},
            lexical_score=0.8,
            semantic_score=0.8,
            novelty_score=0.5,
            cluster_id="cluster-a",
        ),
        ContextCandidate(
            id="b",
            lane=RetrievalLane.CORPUS,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="beta rollback process denied",
            tokens=20,
            provenance={"source": "corpus"},
            lexical_score=0.8,
            semantic_score=0.8,
            novelty_score=0.5,
            cluster_id="cluster-b",
        ),
    ]

    class _Adapter:
        def contradiction_score(self, left: str, right: str) -> float:
            if left.strip().lower() == "alpha release checklist approved" and right.strip().lower() == "beta rollback process denied":
                return 0.91
            return 0.10

    monkeypatch.setenv("CRAG_COMPAT_ENGINE", "nli")
    monkeypatch.setenv("CRAG_COMPAT_NLI_BACKEND", "transformers")
    monkeypatch.setattr(
        "CognitiveRAG.crag.context_selection.compatibility.TransformersNLIAdapter",
        lambda model_name=None: _Adapter(),
    )
    monkeypatch.setattr(
        "CognitiveRAG.session_memory.context_window.build_candidates_with_route",
        lambda **_: (route_plan, lane_candidates),
    )
    monkeypatch.setattr(
        "CognitiveRAG.session_memory.context_window.DiscoveryExecutor.run",
        lambda self, **_: DiscoveryResult(budget_tokens=220, used_tokens=0, injected_discoveries=[]),
    )

    out = assemble_context(
        session_id,
        fresh_tail_count=0,
        budget=320,
        query="investigate conflicting evidence",
        intent_family=IntentFamily.INVESTIGATIVE,
    )

    selected_ids = [block["id"] for block in out["selected_blocks"]]
    dropped = {block["id"]: block["reason"] for block in out["dropped_blocks"]}
    runtime_state = out["selector_metrics"]["decision_stats"]["compatibility_engine"]

    assert selected_ids == ["a"]
    assert dropped.get("b") == "compatibility_conflict_nli"
    assert runtime_state["resolved_engine"] == "nli"
    assert runtime_state["backend_available"] is True
    assert runtime_state["fallback_active"] is False
