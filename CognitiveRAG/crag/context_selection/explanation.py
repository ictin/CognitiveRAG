from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.contracts.schemas import DroppedBlock, SelectionExplanation, SelectedBlock
from CognitiveRAG.crag.contracts.schemas import ContextCandidate


def build_explanation(
    *,
    intent_family: IntentFamily,
    total_budget: int,
    reserved_tokens: int,
    selected: Iterable[tuple[ContextCandidate, float]],
    dropped: Iterable[tuple[ContextCandidate, str]],
    reorder_strategy: str = "front_back_anchor",
) -> SelectionExplanation:
    lane_totals: dict[str, int] = defaultdict(int)
    selected_blocks: list[SelectedBlock] = []
    cluster_coverage: set[str] = set()

    for candidate, utility in selected:
        lane_totals[candidate.lane.value] += candidate.tokens
        if candidate.cluster_id:
            cluster_coverage.add(candidate.cluster_id)
        selected_blocks.append(
            SelectedBlock(
                id=candidate.id,
                lane=candidate.lane.value,
                memory_type=candidate.memory_type.value,
                tokens=candidate.tokens,
                utility=round(float(utility), 6),
                contradiction_risk=round(float(candidate.contradiction_risk), 6),
                cluster_id=candidate.cluster_id,
                provenance=candidate.provenance,
            )
        )

    dropped_blocks = [
        DroppedBlock(
            id=c.id,
            lane=c.lane.value,
            tokens=c.tokens,
            reason=reason,
            contradiction_risk=round(float(c.contradiction_risk), 6),
        )
        for c, reason in dropped
    ]

    return SelectionExplanation(
        intent_family=intent_family,
        total_budget=int(total_budget),
        reserved_tokens=int(reserved_tokens),
        selected_blocks=selected_blocks,
        dropped_blocks=dropped_blocks,
        lane_totals=dict(lane_totals),
        cluster_coverage=sorted(cluster_coverage),
        reorder_strategy=reorder_strategy,
    )
