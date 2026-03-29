from __future__ import annotations

from typing import Iterable, List

from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import RoleProbe


_ROLE_ORDER = {
    IntentFamily.EXACT_RECALL: ['skeptic', 'ops reviewer'],
    IntentFamily.MEMORY_SUMMARY: ['architect', 'skeptic', 'domain specialist'],
    IntentFamily.ARCHITECTURE_EXPLANATION: ['architect', 'security auditor'],
    IntentFamily.CORPUS_OVERVIEW: ['domain specialist', 'skeptic', 'compliance reviewer'],
    IntentFamily.PLANNING: ['ops reviewer', 'architect', 'security auditor'],
    IntentFamily.INVESTIGATIVE: ['skeptic', 'domain specialist', 'security auditor', 'ops reviewer'],
}


def _lane_names(lanes: Iterable[RetrievalLane]) -> str:
    return ', '.join(lane.value for lane in lanes)


def _probe(
    *,
    role: str,
    query: str,
    purpose: str,
    expected_lanes: list[RetrievalLane],
    priority: int,
) -> RoleProbe:
    prompt = f'[{role}] {purpose} Query: {query.strip() or "(empty query)"}. Target lanes: {_lane_names(expected_lanes)}.'
    return RoleProbe(
        role=role,
        prompt=prompt,
        purpose=purpose,
        expected_lanes=expected_lanes,
        priority=priority,
    )


def build_role_conditioned_probes(
    *,
    intent_family: IntentFamily,
    query: str,
    expected_lanes: list[RetrievalLane],
    max_probes: int = 4,
) -> List[RoleProbe]:
    roles = _ROLE_ORDER.get(intent_family, _ROLE_ORDER[IntentFamily.INVESTIGATIVE])
    bounded_roles = roles[: max(1, max_probes)]
    probes: List[RoleProbe] = []
    for idx, role in enumerate(bounded_roles):
        probes.append(
            _probe(
                role=role,
                query=query,
                purpose='Generate a bounded retrieval probe from this role perspective.',
                expected_lanes=expected_lanes,
                priority=idx + 1,
            )
        )
    return probes


def build_contradiction_probes(
    *,
    intent_family: IntentFamily,
    query: str,
    expected_lanes: list[RetrievalLane],
    web_sensitive: bool,
) -> List[RoleProbe]:
    probes: List[RoleProbe] = []
    probes.append(
        _probe(
            role='skeptic',
            query=query,
            purpose='Find evidence that conflicts with the current strongest claim.',
            expected_lanes=expected_lanes,
            priority=1,
        )
    )
    if intent_family in {IntentFamily.ARCHITECTURE_EXPLANATION, IntentFamily.PLANNING}:
        probes.append(
            _probe(
                role='security auditor',
                query=query,
                purpose='Check for policy/safety contradictions in architecture or process claims.',
                expected_lanes=expected_lanes,
                priority=2,
            )
        )
    if web_sensitive:
        probes.append(
            _probe(
                role='compliance reviewer',
                query=query,
                purpose='Cross-check freshness-sensitive claims against recent external evidence.',
                expected_lanes=expected_lanes,
                priority=3,
            )
        )
    return probes[:4]


def build_novelty_probes(
    *,
    intent_family: IntentFamily,
    query: str,
    expected_lanes: list[RetrievalLane],
) -> List[RoleProbe]:
    if intent_family == IntentFamily.ARCHITECTURE_EXPLANATION:
        return []

    probes: List[RoleProbe] = [
        _probe(
            role='domain specialist',
            query=query,
            purpose='Find one high-value angle not already captured in obvious evidence.',
            expected_lanes=expected_lanes,
            priority=1,
        )
    ]
    if intent_family in {IntentFamily.CORPUS_OVERVIEW, IntentFamily.INVESTIGATIVE, IntentFamily.PLANNING}:
        probes.append(
            _probe(
                role='ops reviewer',
                query=query,
                purpose='Find practical edge-cases or operational implications missing from the baseline view.',
                expected_lanes=expected_lanes,
                priority=2,
            )
        )
    return probes[:3]
