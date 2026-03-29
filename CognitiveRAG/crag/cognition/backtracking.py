from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class BranchCandidate:
    branch_id: str
    query: str
    score: float
    evidence_count: int


@dataclass(frozen=True)
class BacktrackingPolicy:
    min_branch_score: float = 0.22
    min_evidence_count: int = 1
    max_branches: int = 4


def should_backtrack(branch: BranchCandidate, policy: BacktrackingPolicy) -> tuple[bool, str | None]:
    if branch.score < policy.min_branch_score:
        return True, 'low_score'
    if branch.evidence_count < policy.min_evidence_count:
        return True, 'insufficient_evidence'
    return False, None


def pick_branch_order(branches: Iterable[BranchCandidate]) -> List[BranchCandidate]:
    return sorted(list(branches), key=lambda b: (-b.score, -b.evidence_count, b.branch_id))


def execute_backtracking(
    branches: Iterable[BranchCandidate],
    policy: BacktrackingPolicy,
) -> Tuple[List[BranchCandidate], List[tuple[BranchCandidate, str]]]:
    explored: List[BranchCandidate] = []
    rejected: List[tuple[BranchCandidate, str]] = []

    ordered = pick_branch_order(branches)
    for branch in ordered:
        backtrack, reason = should_backtrack(branch, policy)
        if backtrack:
            rejected.append((branch, reason or 'backtracked'))
            continue
        if len(explored) >= policy.max_branches:
            rejected.append((branch, 'branch_budget_exceeded'))
            continue
        explored.append(branch)

    return explored, rejected
