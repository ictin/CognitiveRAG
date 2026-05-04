from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.explanation import build_explanation


def test_explanation_contains_lane_totals_selected_and_dropped():
    c1 = ContextCandidate(id="x", lane=RetrievalLane.CORPUS, memory_type=MemoryType.CORPUS_CHUNK, text="hello", tokens=12, provenance={})
    c2 = ContextCandidate(id="y", lane=RetrievalLane.EPISODIC, memory_type=MemoryType.EPISODIC_RAW, text="world", tokens=8, provenance={})

    ex = build_explanation(
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        total_budget=200,
        reserved_tokens=40,
        selected=[(c1, 1.2)],
        dropped=[(c2, "budget_or_lane_cap")],
    )

    assert ex.lane_totals["corpus"] == 12
    assert ex.selected_blocks[0].id == "x"
    assert ex.dropped_blocks[0].id == "y"
    assert ex.total_budget == 200
    assert ex.reserved_tokens == 40


def test_explanation_preserves_lifecycle_truthfulness_in_provenance():
    c1 = ContextCandidate(
        id="wp_stale",
        lane=RetrievalLane.WEB,
        memory_type=MemoryType.WEB_PROMOTED_FACT,
        text="Aging promoted claim",
        tokens=16,
        provenance={
            "source_class": "web_promoted",
            "lifecycle": {
                "lifecycle_state": "revalidation_required",
                "approval_state": "approved",
                "freshness_lifecycle_state": "revalidation_pending",
                "revalidation_state": "required",
                "revalidation_requested": True,
                "validated": True,
            },
        },
    )
    ex = build_explanation(
        intent_family=IntentFamily.INVESTIGATIVE,
        total_budget=220,
        reserved_tokens=40,
        selected=[(c1, 1.05)],
        dropped=[],
    )
    prov = dict(ex.selected_blocks[0].provenance or {})
    lifecycle = dict(prov.get("lifecycle") or {})
    assert lifecycle.get("lifecycle_state") == "revalidation_required"
    assert lifecycle.get("revalidation_state") == "required"
    # Truthfulness guardrail: explanation must not imply revalidated when pending.
    assert lifecycle.get("lifecycle_state") != "revalidated"
