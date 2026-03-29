from __future__ import annotations

from typing import Iterable, List

from CognitiveRAG.crag.contracts.schemas import ContextCandidate, ContextSelectionPolicy


def reorder_for_prompt(
    selected: Iterable[tuple[ContextCandidate, float]],
    policy: ContextSelectionPolicy,
) -> List[tuple[ContextCandidate, float]]:
    """LongContextReorder-like deterministic placement.

    High utility and must_include items are anchored to front/back buckets.
    """
    ordered = sorted(
        list(selected),
        key=lambda x: (
            0 if x[0].must_include else 1,
            -x[1],
            x[0].id,
        ),
    )
    if not ordered:
        return []

    front_n = min(policy.front_anchor_budget, len(ordered))
    back_n = min(policy.back_anchor_budget, max(0, len(ordered) - front_n))

    front = ordered[:front_n]
    middle = ordered[front_n : len(ordered) - back_n]
    back = ordered[len(ordered) - back_n :] if back_n > 0 else []

    # Keep middle deterministic but less aggressive than anchors.
    middle = sorted(middle, key=lambda x: (-x[1], x[0].id))
    back = sorted(back, key=lambda x: (-x[1], x[0].id))

    return front + middle + back
