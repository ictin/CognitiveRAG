from CognitiveRAG.crag.cognition.probes import (
    build_contradiction_probes,
    build_novelty_probes,
    build_role_conditioned_probes,
)
from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane


LANES = [RetrievalLane.SEMANTIC, RetrievalLane.CORPUS, RetrievalLane.WEB]


def test_role_conditioned_probes_are_deterministic():
    p1 = build_role_conditioned_probes(
        intent_family=IntentFamily.INVESTIGATIVE,
        query='Investigate contradictions in the rollout plan',
        expected_lanes=LANES,
        max_probes=3,
    )
    p2 = build_role_conditioned_probes(
        intent_family=IntentFamily.INVESTIGATIVE,
        query='Investigate contradictions in the rollout plan',
        expected_lanes=LANES,
        max_probes=3,
    )

    assert [probe.model_dump(mode='json') for probe in p1] == [probe.model_dump(mode='json') for probe in p2]
    assert [probe.role for probe in p1] == ['skeptic', 'domain specialist', 'security auditor']


def test_contradiction_and_novelty_probe_shapes():
    contradiction = build_contradiction_probes(
        intent_family=IntentFamily.PLANNING,
        query='Plan a secure release and verify constraints',
        expected_lanes=LANES,
        web_sensitive=True,
    )
    novelty = build_novelty_probes(
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        query='What does this book say about retention?',
        expected_lanes=LANES,
    )

    assert len(contradiction) >= 2
    assert contradiction[0].role == 'skeptic'
    assert all(probe.expected_lanes for probe in contradiction)
    assert len(novelty) >= 1
    assert novelty[0].role == 'domain specialist'
