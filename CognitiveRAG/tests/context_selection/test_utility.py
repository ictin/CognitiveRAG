from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.utility import score_candidate


def _cand(cid: str, lane: RetrievalLane, mtype: MemoryType, lex: float = 0.7):
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=mtype,
        text=f"text {cid}",
        tokens=20,
        provenance={"src": "t"},
        lexical_score=lex,
        semantic_score=lex,
        recency_score=0.5,
        freshness_score=0.5,
        trust_score=0.7,
        novelty_score=0.6,
        contradiction_risk=0.0,
    )


def test_intent_changes_scoring_priority():
    episodic = _cand("e1", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW)
    summary = _cand("s1", RetrievalLane.SESSION_SUMMARY, MemoryType.SUMMARY)

    exact_score_episodic = score_candidate(episodic, [], get_policy(IntentFamily.EXACT_RECALL), IntentFamily.EXACT_RECALL, False)
    exact_score_summary = score_candidate(summary, [], get_policy(IntentFamily.EXACT_RECALL), IntentFamily.EXACT_RECALL, False)
    assert exact_score_episodic > exact_score_summary

    mem_score_summary = score_candidate(summary, [], get_policy(IntentFamily.MEMORY_SUMMARY), IntentFamily.MEMORY_SUMMARY, False)
    mem_score_episodic = score_candidate(episodic, [], get_policy(IntentFamily.MEMORY_SUMMARY), IntentFamily.MEMORY_SUMMARY, False)
    assert mem_score_summary > mem_score_episodic


def test_corpus_overview_prefers_corpus_lane_when_relevant():
    corpus = _cand("c1", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, lex=0.8)
    episodic = _cand("e2", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW, lex=0.8)
    policy = get_policy(IntentFamily.CORPUS_OVERVIEW)
    s_c = score_candidate(corpus, [], policy, IntentFamily.CORPUS_OVERVIEW, False)
    s_e = score_candidate(episodic, [], policy, IntentFamily.CORPUS_OVERVIEW, False)
    assert s_c > s_e
