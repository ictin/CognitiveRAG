from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from CognitiveRAG.crag.contracts.schemas import (
    ContradictionRecord,
    DiscoveryBranchRecord,
    DiscoveryEvidenceRef,
    DiscoveryLedgerSnapshot,
)


@dataclass
class GlobalLedger:
    explored_branches: List[DiscoveryBranchRecord] = field(default_factory=list)
    rejected_branches: List[DiscoveryBranchRecord] = field(default_factory=list)
    evidence_bundles: Dict[str, List[str]] = field(default_factory=dict)
    contradictions: List[ContradictionRecord] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    weak_branch_abandonments: List[str] = field(default_factory=list)
    next_branch_suggestions: List[str] = field(default_factory=list)

    def record_explored(self, branch: DiscoveryBranchRecord, evidence: List[DiscoveryEvidenceRef]) -> None:
        self.explored_branches.append(branch)
        self.evidence_bundles[branch.branch_id] = [item.evidence_id for item in evidence]

    def record_rejected(self, branch: DiscoveryBranchRecord) -> None:
        self.rejected_branches.append(branch)
        if branch.abandoned and branch.branch_id not in self.weak_branch_abandonments:
            self.weak_branch_abandonments.append(branch.branch_id)

    def add_contradictions(self, rows: List[ContradictionRecord]) -> None:
        self.contradictions.extend(rows)

    def add_unresolved(self, question: str) -> None:
        text = str(question or '').strip()
        if not text:
            return
        if text not in self.unresolved_questions:
            self.unresolved_questions.append(text)

    def add_next_suggestion(self, suggestion: str) -> None:
        text = str(suggestion or "").strip()
        if not text:
            return
        if text not in self.next_branch_suggestions:
            self.next_branch_suggestions.append(text)

    def snapshot(self) -> DiscoveryLedgerSnapshot:
        return DiscoveryLedgerSnapshot(
            explored_branches=list(self.explored_branches),
            rejected_branches=list(self.rejected_branches),
            evidence_bundles={k: list(v) for k, v in sorted(self.evidence_bundles.items())},
            contradictions=list(self.contradictions),
            unresolved_questions=list(self.unresolved_questions),
            weak_branch_abandonments=list(self.weak_branch_abandonments),
            next_branch_suggestions=list(self.next_branch_suggestions),
        )
