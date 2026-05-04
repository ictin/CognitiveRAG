from __future__ import annotations

from dataclasses import dataclass
from typing import List

from CognitiveRAG.crag.cognition.backtracking import BacktrackingPolicy, BranchCandidate, execute_backtracking
from CognitiveRAG.crag.cognition.contradiction import detect_contradictions
from CognitiveRAG.crag.cognition.curiosity import score_candidate_curiosity, update_seen_tokens
from CognitiveRAG.crag.cognition.ledger import GlobalLedger
from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import (
    ContextCandidate,
    DiscoveryBranchRecord,
    DiscoveryEvidenceRef,
    DiscoveryPlan,
    DiscoveryResult,
    InjectedDiscovery,
)


@dataclass(frozen=True)
class DiscoveryPolicy:
    max_branches: int = 4
    max_evidence_per_branch: int = 3
    min_branch_score: float = 0.22
    min_evidence_count: int = 1
    injection_budget_tokens: int = 220
    max_injected_discoveries: int = 3


class DiscoveryExecutor:
    """Bounded discovery executor for M10 (no graph/federation)."""

    def __init__(self, policy: DiscoveryPolicy | None = None):
        self.policy = policy or DiscoveryPolicy()

    def run(self, *, plan: DiscoveryPlan, candidate_pool: List[ContextCandidate]) -> DiscoveryResult:
        ledger = GlobalLedger()
        seen_tokens = set()

        branch_rows: list[tuple[BranchCandidate, list[DiscoveryEvidenceRef]]] = []
        # Execute the full bounded probe set from planning:
        # role-conditioned + contradiction + novelty.
        probes = [
            *list(plan.role_conditioned_probes or []),
            *list(plan.contradiction_probes or []),
            *list(plan.novelty_probes or []),
        ]

        for idx, probe in enumerate(probes[: self.policy.max_branches * 3]):
            branch_id = f'branch-{idx + 1:02d}'
            scored: list[tuple[float, ContextCandidate]] = []
            for cand in candidate_pool:
                if probe.expected_lanes and cand.lane not in probe.expected_lanes:
                    continue
                score = score_candidate_curiosity(cand, query=probe.prompt, seen_tokens=seen_tokens)
                if score <= 0.01:
                    continue
                scored.append((score, cand))

            scored.sort(key=lambda row: (-row[0], row[1].id))
            top = scored[: self.policy.max_evidence_per_branch]

            evidence_rows: list[DiscoveryEvidenceRef] = []
            for rank, (score, cand) in enumerate(top):
                evidence_rows.append(
                    DiscoveryEvidenceRef(
                        evidence_id=f'{branch_id}:{cand.id}',
                        branch_id=branch_id,
                        lane=cand.lane,
                        memory_type=cand.memory_type,
                        text=cand.text,
                        tokens=cand.tokens,
                        score=score,
                        provenance=dict(cand.provenance or {}),
                    )
                )

            branch_score = round(sum(item.score for item in evidence_rows) / float(len(evidence_rows) or 1), 6)
            branch_rows.append(
                (
                    BranchCandidate(
                        branch_id=branch_id,
                        query=probe.prompt,
                        score=branch_score,
                        evidence_count=len(evidence_rows),
                    ),
                    evidence_rows,
                )
            )
            seen_tokens = update_seen_tokens(seen_tokens, [item.text for item in evidence_rows])

        explored, rejected = execute_backtracking(
            [row[0] for row in branch_rows],
            BacktrackingPolicy(
                min_branch_score=self.policy.min_branch_score,
                min_evidence_count=self.policy.min_evidence_count,
                max_branches=self.policy.max_branches,
            ),
        )

        evidence_by_branch = {branch.branch_id: evidence for branch, evidence in branch_rows}
        explored_evidence: list[DiscoveryEvidenceRef] = []

        for branch in explored:
            evidence = evidence_by_branch.get(branch.branch_id, [])
            explored_evidence.extend(evidence)
            ledger.record_explored(
                DiscoveryBranchRecord(
                    branch_id=branch.branch_id,
                    query=branch.query,
                    status='explored',
                    score=branch.score,
                    evidence_ids=[item.evidence_id for item in evidence],
                ),
                evidence,
            )

        for branch, reason in rejected:
            ledger.record_rejected(
                DiscoveryBranchRecord(
                    branch_id=branch.branch_id,
                    query=branch.query,
                    status='rejected',
                    score=branch.score,
                    evidence_ids=[item.evidence_id for item in evidence_by_branch.get(branch.branch_id, [])],
                    reason=reason,
                )
            )

        # Contradictions are first-class signals and should be surfaced from the
        # bounded discovery evidence set even if some branches are later rejected.
        bounded_evidence = [row for _, evidence in branch_rows for row in evidence]
        contradictions = detect_contradictions(bounded_evidence)
        if contradictions:
            ledger.add_contradictions(contradictions)

        if not explored_evidence:
            ledger.add_unresolved('No discovery evidence passed branch quality thresholds.')

        injected: list[InjectedDiscovery] = []
        used_tokens = 0
        seen_discovery_ids = set()
        ranked_evidence = sorted(explored_evidence, key=lambda item: (-item.score, item.evidence_id))
        for evidence in ranked_evidence:
            if len(injected) >= self.policy.max_injected_discoveries:
                break
            if evidence.evidence_id in seen_discovery_ids:
                continue
            if used_tokens + evidence.tokens > self.policy.injection_budget_tokens:
                continue
            injected.append(
                InjectedDiscovery(
                    discovery_id=f'discovery:{evidence.evidence_id}',
                    text=evidence.text,
                    tokens=evidence.tokens,
                    score=evidence.score,
                    source_evidence_ids=[evidence.evidence_id],
                    provenance={
                        **(evidence.provenance or {}),
                        'branch_id': evidence.branch_id,
                        'lane': evidence.lane.value,
                        'memory_type': evidence.memory_type.value,
                    },
                )
            )
            seen_discovery_ids.add(evidence.evidence_id)
            used_tokens += evidence.tokens

        return DiscoveryResult(
            bounded=True,
            budget_tokens=self.policy.injection_budget_tokens,
            used_tokens=used_tokens,
            injected_discoveries=injected,
            contradictions=contradictions,
            ledger=ledger.snapshot(),
        )


def discovery_items_to_candidates(items: List[InjectedDiscovery]) -> List[ContextCandidate]:
    out: List[ContextCandidate] = []
    for item in items:
        out.append(
            ContextCandidate(
                id=item.discovery_id,
                lane=RetrievalLane.DISCOVERY,
                memory_type=MemoryType.SUMMARY,
                text=item.text,
                tokens=item.tokens,
                provenance=dict(item.provenance or {}),
                lexical_score=0.0,
                semantic_score=min(1.0, item.score),
                recency_score=0.4,
                freshness_score=0.5,
                trust_score=0.6,
                novelty_score=min(1.0, item.score),
                contradiction_risk=0.0,
                cluster_id='discovery',
                must_include=False,
                compressible=True,
            )
        )
    return out
