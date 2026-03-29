from CognitiveRAG.crag.cognition.curiosity import novelty_score, score_candidate_curiosity, uncertainty_score
from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate


def test_novelty_and_uncertainty_scores_are_deterministic():
    seen = {'known', 'token'}
    assert novelty_score(text='known token newvalue', seen_tokens=seen) == 1 / 3
    assert uncertainty_score('this is maybe uncertain') > 0


def test_curiosity_scoring_rewards_query_overlap_and_novelty():
    candidate = ContextCandidate(
        id='c1',
        lane=RetrievalLane.SEMANTIC,
        memory_type=MemoryType.SUMMARY,
        text='migration rollout risk mitigation checklist',
        contradiction_risk=0.2,
    )
    s1 = score_candidate_curiosity(candidate, query='migration risk', seen_tokens={'old'})
    s2 = score_candidate_curiosity(candidate, query='unrelated topic', seen_tokens={'migration', 'rollout', 'risk', 'mitigation', 'checklist'})
    assert s1 > s2
