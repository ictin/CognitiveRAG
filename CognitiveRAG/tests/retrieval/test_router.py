from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.retrieval.router import LaneRouter


def test_router_exact_recall_prefers_episodic():
    plan = LaneRouter().route(query="what did we say earlier about X", intent_family=IntentFamily.EXACT_RECALL)
    assert plan.lanes[0] == RetrievalLane.EPISODIC
    assert RetrievalLane.CORPUS not in plan.lanes[:2]


def test_router_memory_summary_prefers_promoted():
    plan = LaneRouter().route(query="what do you know about me", intent_family=IntentFamily.MEMORY_SUMMARY)
    assert plan.lanes[0] == RetrievalLane.PROMOTED


def test_router_corpus_prefers_corpus_and_large_file():
    plan = LaneRouter().route(query="what does this book say", intent_family=IntentFamily.CORPUS_OVERVIEW)
    assert plan.lanes[:2] == [RetrievalLane.CORPUS, RetrievalLane.LARGE_FILE]


def test_router_mixed_investigative_has_mixed_lanes():
    plan = LaneRouter().route(query="mix recall and corpus", intent_family=IntentFamily.INVESTIGATIVE)
    assert RetrievalLane.EPISODIC in plan.lanes
    assert RetrievalLane.CORPUS in plan.lanes
    assert RetrievalLane.SEMANTIC in plan.lanes
