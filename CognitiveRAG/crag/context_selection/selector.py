from __future__ import annotations

from typing import Iterable, List, Tuple

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, ContextSelectionPolicy
from CognitiveRAG.crag.context_selection.compatibility import (
    CompatibilityEngine,
    compatibility_conflict_reason,
    resolve_compatibility_engine,
)
from CognitiveRAG.crag.context_selection.explanation import build_explanation
from CognitiveRAG.crag.context_selection.models import SelectionState
from CognitiveRAG.crag.context_selection.reorder import reorder_for_prompt
from CognitiveRAG.crag.context_selection.utility import score_candidate


def _max_for_lane(policy: ContextSelectionPolicy, lane: str) -> int:
    return int(policy.lane_maxima.get(lane, 999_999))


def _min_for_lane(policy: ContextSelectionPolicy, lane: str) -> int:
    return int(policy.lane_minima.get(lane, 0))


def _can_add(candidate: ContextCandidate, state: SelectionState, policy: ContextSelectionPolicy, total_budget: int) -> bool:
    if state.used_tokens + candidate.tokens > total_budget:
        return False
    lane = candidate.lane.value
    if state.lane_counts.get(lane, 0) >= _max_for_lane(policy, lane):
        return False
    return True


def select_context(
    *,
    candidates: Iterable[ContextCandidate],
    policy: ContextSelectionPolicy,
    total_budget: int,
    reserved_tokens: int,
    intent_family: IntentFamily,
    contradiction_threshold: float = 0.95,
    pairwise_compatibility: bool = True,
    compatibility_engine: CompatibilityEngine | None = None,
) -> tuple[list[tuple[ContextCandidate, float]], list[tuple[ContextCandidate, str]], object]:
    """Budgeted multi-store context optimizer.

    Returns (selected_with_utility, dropped_with_reason, explanation_model).
    """
    all_candidates: List[ContextCandidate] = list(candidates)
    state = SelectionState(used_tokens=int(reserved_tokens))
    dropped: List[Tuple[ContextCandidate, str]] = []

    # Phase F prefilter: deterministic contradiction guard.
    filtered: List[ContextCandidate] = []
    for candidate in all_candidates:
        if candidate.contradiction_risk > contradiction_threshold:
            dropped.append((candidate, "contradiction_risk"))
            continue
        filtered.append(candidate)

    # Phase A/B selected must_include first.
    scored_map: dict[str, float] = {}
    selected_clusters: set[str] = set()
    active_compatibility_engine = compatibility_engine or resolve_compatibility_engine(mode="heuristic")
    must_include = [c for c in filtered if c.must_include]
    for candidate in sorted(must_include, key=lambda c: (c.tokens, c.id)):
        if pairwise_compatibility:
            reason = compatibility_conflict_reason(candidate, state.selected, engine=active_compatibility_engine)
            if reason:
                dropped.append((candidate, reason))
                continue
        if not _can_add(candidate, state, policy, total_budget):
            dropped.append((candidate, "budget_or_lane_cap"))
            continue
        util = score_candidate(
            candidate,
            state.selected,
            policy,
            intent_family,
            cluster_unseen_bonus=bool(candidate.cluster_id and candidate.cluster_id not in selected_clusters),
        )
        state.add(candidate)
        scored_map[candidate.id] = util
        if candidate.cluster_id:
            selected_clusters.add(candidate.cluster_id)

    # Phase E.1: satisfy lane minima using best value-per-token first.
    remaining = [c for c in filtered if c.id not in state.selected_ids]
    for lane, minimum in sorted(policy.lane_minima.items()):
        while state.lane_counts.get(lane, 0) < minimum:
            lane_candidates = [c for c in remaining if c.lane.value == lane]
            if not lane_candidates:
                break
            scored = []
            for candidate in lane_candidates:
                util = score_candidate(
                    candidate,
                    state.selected,
                    policy,
                    intent_family,
                    cluster_unseen_bonus=bool(candidate.cluster_id and candidate.cluster_id not in selected_clusters),
                )
                value_per_token = util / float(max(1, candidate.tokens))
                scored.append((value_per_token, util, candidate))
            scored.sort(key=lambda x: (-x[0], -x[1], x[2].id))
            _, util, best = scored[0]
            if pairwise_compatibility:
                reason = compatibility_conflict_reason(best, state.selected, engine=active_compatibility_engine)
                if reason:
                    dropped.append((best, reason))
                    remaining = [c for c in remaining if c.id != best.id]
                    continue
            if not _can_add(best, state, policy, total_budget):
                dropped.append((best, "budget_or_lane_cap"))
                remaining = [c for c in remaining if c.id != best.id]
                continue
            state.add(best)
            scored_map[best.id] = util
            if best.cluster_id:
                selected_clusters.add(best.cluster_id)
            remaining = [c for c in remaining if c.id != best.id]

    # Phase E.2: global greedy by marginal utility/token.
    while remaining:
        scored = []
        for candidate in remaining:
            util = score_candidate(
                candidate,
                state.selected,
                policy,
                intent_family,
                cluster_unseen_bonus=bool(candidate.cluster_id and candidate.cluster_id not in selected_clusters),
            )
            scored.append((util / float(max(1, candidate.tokens)), util, candidate))

        scored.sort(key=lambda x: (-x[0], -x[1], x[2].id))
        _, util, best = scored[0]
        remaining = [c for c in remaining if c.id != best.id]

        if pairwise_compatibility:
            reason = compatibility_conflict_reason(best, state.selected, engine=active_compatibility_engine)
            if reason:
                dropped.append((best, reason))
                continue

        if not _can_add(best, state, policy, total_budget):
            dropped.append((best, "budget_or_lane_cap"))
            continue

        state.add(best)
        scored_map[best.id] = util
        if best.cluster_id:
            selected_clusters.add(best.cluster_id)

    selected_pairs = [(c, scored_map.get(c.id, 0.0)) for c in state.selected]

    # Phase G: prompt ordering anchors.
    selected_pairs = reorder_for_prompt(selected_pairs, policy)

    # Phase H: machine-readable explanation.
    explanation = build_explanation(
        intent_family=intent_family,
        total_budget=total_budget,
        reserved_tokens=reserved_tokens,
        selected=selected_pairs,
        dropped=dropped,
    )

    return selected_pairs, dropped, explanation
