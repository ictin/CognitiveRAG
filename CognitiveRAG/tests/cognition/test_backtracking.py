from CognitiveRAG.crag.cognition.backtracking import (
    BacktrackingPolicy,
    BranchCandidate,
    execute_backtracking,
)


def test_weak_branch_is_backtracked_and_alternative_explored():
    branches = [
        BranchCandidate(branch_id='b1', query='weak', score=0.1, evidence_count=2),
        BranchCandidate(branch_id='b2', query='strong', score=0.7, evidence_count=2),
        BranchCandidate(branch_id='b3', query='also-strong', score=0.5, evidence_count=1),
    ]
    explored, rejected = execute_backtracking(branches, BacktrackingPolicy(min_branch_score=0.2, max_branches=2))

    assert [b.branch_id for b in explored] == ['b2', 'b3']
    assert any(b.branch_id == 'b1' and reason == 'low_score' for b, reason in rejected)
