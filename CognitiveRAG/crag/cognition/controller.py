from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from CognitiveRAG.crag.cognition.probes import (
    build_contradiction_probes,
    build_novelty_probes,
    build_role_conditioned_probes,
)
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import DiscoveryPlan


ControllerMode = Literal['rule_based', 'mock']


@dataclass(frozen=True)
class ControllerSettings:
    mode: ControllerMode = 'rule_based'
    max_query_variants: int = 4
    max_adjacent_topics: int = 6


class CognitiveController:
    """M9 structured planner (controller only, no discovery execution)."""

    def __init__(self, settings: ControllerSettings | None = None):
        self.settings = settings or ControllerSettings()

    @property
    def mode(self) -> ControllerMode:
        return self.settings.mode

    @classmethod
    def mock(cls) -> 'CognitiveController':
        return cls(ControllerSettings(mode='mock'))

    def build_plan(
        self,
        *,
        query: str,
        hinted_intent: IntentFamily | str | None = None,
        local_evidence_count: int = 0,
        available_lanes: Iterable[RetrievalLane] | None = None,
    ) -> DiscoveryPlan:
        normalized_query = self._normalize_query(query)
        intent = self._resolve_intent(normalized_query, hinted_intent)
        web_sensitive = self._is_web_sensitive(normalized_query)
        expected_lanes = self._expected_lanes(intent, web_sensitive, available_lanes)

        if self.mode == 'mock':
            return self._build_mock_plan(
                query=normalized_query,
                intent=intent,
                web_sensitive=web_sensitive,
                expected_lanes=expected_lanes,
            )

        discovery_mode = self._discovery_mode(intent, web_sensitive, local_evidence_count)
        risk_mode = self._risk_mode(normalized_query)
        query_variants = self._query_variants(normalized_query)
        adjacent_topics = self._adjacent_topics(normalized_query)

        role_probes = build_role_conditioned_probes(
            intent_family=intent,
            query=normalized_query,
            expected_lanes=expected_lanes,
            max_probes=4,
        )
        contradiction_probes = build_contradiction_probes(
            intent_family=intent,
            query=normalized_query,
            expected_lanes=expected_lanes,
            web_sensitive=web_sensitive,
        )
        novelty_probes = build_novelty_probes(
            intent_family=intent,
            query=normalized_query,
            expected_lanes=expected_lanes,
        )

        notes = [
            'm9-controller: structured planning only',
            'm10-discovery-loop not executed here',
        ]
        if web_sensitive:
            notes.append('web-sensitivity detected from query terms')
        if hinted_intent:
            notes.append('intent hint considered')

        return DiscoveryPlan(
            intent_family=intent,
            discovery_mode=discovery_mode,
            web_sensitive=web_sensitive,
            risk_mode=risk_mode,
            bounded=True,
            query_variants=query_variants,
            adjacent_topics=adjacent_topics,
            expected_lanes=expected_lanes,
            role_conditioned_probes=role_probes,
            contradiction_probes=contradiction_probes,
            novelty_probes=novelty_probes,
            notes=notes,
        )

    def _build_mock_plan(
        self,
        *,
        query: str,
        intent: IntentFamily,
        web_sensitive: bool,
        expected_lanes: list[RetrievalLane],
    ) -> DiscoveryPlan:
        role_probes = build_role_conditioned_probes(
            intent_family=intent,
            query=query,
            expected_lanes=expected_lanes,
            max_probes=2,
        )
        return DiscoveryPlan(
            plan_version='m9-mock',
            intent_family=intent,
            discovery_mode=DiscoveryMode.PASSIVE if intent == IntentFamily.INVESTIGATIVE else DiscoveryMode.OFF,
            web_sensitive=web_sensitive,
            risk_mode='mock',
            bounded=True,
            query_variants=[query or '(empty query)', f'mock:{intent.value}'],
            adjacent_topics=['mock-topic-a', 'mock-topic-b'],
            expected_lanes=expected_lanes,
            role_conditioned_probes=role_probes,
            contradiction_probes=[],
            novelty_probes=[],
            notes=['mock-controller-mode', 'deterministic-no-llm'],
        )

    def _normalize_query(self, query: str) -> str:
        return re.sub(r'\s+', ' ', str(query or '')).strip()

    def _resolve_intent(self, query: str, hinted_intent: IntentFamily | str | None) -> IntentFamily:
        if hinted_intent:
            try:
                return IntentFamily(hinted_intent)
            except Exception:
                pass

        q = query.lower()
        if any(token in q for token in ('what did we say', 'earlier', 'quote')):
            return IntentFamily.EXACT_RECALL
        if any(token in q for token in ('investigate', 'investigation', 'unknown unknown', 'what else matters')):
            return IntentFamily.INVESTIGATIVE
        if any(token in q for token in ('how is your memory organized', 'architecture', 'where did this answer come from')):
            return IntentFamily.ARCHITECTURE_EXPLANATION
        if any(token in q for token in ('what do you remember', 'what do you know about me')):
            return IntentFamily.MEMORY_SUMMARY
        if any(token in q for token in ('what can you tell me about', 'synopsis', 'book', 'corpus')):
            return IntentFamily.CORPUS_OVERVIEW
        if any(token in q for token in ('plan', 'next step', 'roadmap', 'sequence')):
            return IntentFamily.PLANNING
        return IntentFamily.INVESTIGATIVE

    def _is_web_sensitive(self, query: str) -> bool:
        q = query.lower()
        freshness_tokens = ('latest', 'current', 'today', 'yesterday', 'recent', 'news', 'update', '2026')
        verification_tokens = ('verify', 'fact-check', 'confirm', 'source', 'citation')
        return any(token in q for token in freshness_tokens + verification_tokens)

    def _expected_lanes(
        self,
        intent: IntentFamily,
        web_sensitive: bool,
        available_lanes: Iterable[RetrievalLane] | None,
    ) -> list[RetrievalLane]:
        by_intent = {
            IntentFamily.EXACT_RECALL: [RetrievalLane.EPISODIC, RetrievalLane.LEXICAL, RetrievalLane.SEMANTIC],
            IntentFamily.MEMORY_SUMMARY: [RetrievalLane.PROMOTED, RetrievalLane.EPISODIC, RetrievalLane.SEMANTIC],
            IntentFamily.ARCHITECTURE_EXPLANATION: [RetrievalLane.PROMOTED, RetrievalLane.EPISODIC],
            IntentFamily.CORPUS_OVERVIEW: [RetrievalLane.CORPUS, RetrievalLane.LARGE_FILE, RetrievalLane.LEXICAL],
            IntentFamily.PLANNING: [RetrievalLane.PROMOTED, RetrievalLane.EPISODIC, RetrievalLane.SEMANTIC],
            IntentFamily.INVESTIGATIVE: [
                RetrievalLane.SEMANTIC,
                RetrievalLane.LEXICAL,
                RetrievalLane.EPISODIC,
                RetrievalLane.PROMOTED,
                RetrievalLane.CORPUS,
                RetrievalLane.LARGE_FILE,
            ],
        }
        lanes = list(by_intent[intent])
        if web_sensitive and RetrievalLane.WEB not in lanes:
            lanes.append(RetrievalLane.WEB)

        if available_lanes is None:
            return lanes

        allowed = set(available_lanes)
        filtered = [lane for lane in lanes if lane in allowed]
        return filtered or lanes

    def _discovery_mode(self, intent: IntentFamily, web_sensitive: bool, local_evidence_count: int) -> DiscoveryMode:
        if intent in {IntentFamily.EXACT_RECALL, IntentFamily.ARCHITECTURE_EXPLANATION}:
            return DiscoveryMode.OFF
        if intent == IntentFamily.INVESTIGATIVE and local_evidence_count <= 0:
            return DiscoveryMode.ACTIVE
        if web_sensitive and local_evidence_count < 2:
            return DiscoveryMode.ACTIVE
        return DiscoveryMode.PASSIVE

    def _risk_mode(self, query: str) -> str:
        q = query.lower()
        if any(token in q for token in ('security', 'compliance', 'legal', 'medical', 'financial')):
            return 'high'
        if any(token in q for token in ('verify', 'contradiction', 'conflict', 'audit')):
            return 'elevated'
        return 'normal'

    def _query_variants(self, query: str) -> list[str]:
        if not query:
            return []
        variants = [query]
        compact = re.sub(r'[^a-z0-9\s]+', ' ', query.lower())
        compact = re.sub(r'\s+', ' ', compact).strip()
        if compact and compact != query.lower():
            variants.append(compact)
        if ' and ' in compact:
            for part in compact.split(' and '):
                p = part.strip()
                if len(p) >= 8:
                    variants.append(p)
        dedup: list[str] = []
        seen = set()
        for variant in variants:
            key = variant.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            dedup.append(variant)
        return dedup[: self.settings.max_query_variants]

    def _adjacent_topics(self, query: str) -> list[str]:
        if not query:
            return []
        tokens = [tok for tok in re.split(r'[^a-z0-9]+', query.lower()) if len(tok) >= 4]
        stop = {
            'what', 'tell', 'about', 'does', 'this', 'that', 'with', 'from', 'into', 'your', 'have', 'been',
            'current', 'latest', 'today', 'recent', 'query', 'where', 'when', 'should', 'would',
        }
        topics = []
        seen = set()
        for tok in tokens:
            if tok in stop:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            topics.append(tok)
        return topics[: self.settings.max_adjacent_topics]
