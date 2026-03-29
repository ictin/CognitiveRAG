from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from CognitiveRAG.crag.contracts.schemas import ContextCandidate


@dataclass
class SelectionState:
    selected: List[ContextCandidate] = field(default_factory=list)
    selected_ids: set[str] = field(default_factory=set)
    lane_counts: Dict[str, int] = field(default_factory=dict)
    used_tokens: int = 0

    def add(self, candidate: ContextCandidate) -> None:
        self.selected.append(candidate)
        self.selected_ids.add(candidate.id)
        self.lane_counts[candidate.lane.value] = self.lane_counts.get(candidate.lane.value, 0) + 1
        self.used_tokens += int(candidate.tokens)
