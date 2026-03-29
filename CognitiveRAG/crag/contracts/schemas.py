from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import IntentFamily, RetrievalLane, MemoryType, DiscoveryMode


class ContextCandidate(BaseModel):
    id: str
    lane: RetrievalLane
    memory_type: MemoryType
    text: str
    tokens: int = 0
    provenance: Dict[str, Any] = Field(default_factory=dict)

    lexical_score: float = 0.0
    semantic_score: float = 0.0
    recency_score: float = 0.0
    freshness_score: float = 0.0
    trust_score: float = 0.0
    novelty_score: float = 0.0
    contradiction_risk: float = 0.0

    cluster_id: str | None = None
    must_include: bool = False
    compressible: bool = True

    @field_validator(
        "lexical_score",
        "semantic_score",
        "recency_score",
        "freshness_score",
        "trust_score",
        "novelty_score",
        "contradiction_risk",
        mode="before",
    )
    @classmethod
    def _coerce_float(cls, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @model_validator(mode="after")
    def _normalize_tokens(self) -> "ContextCandidate":
        if self.tokens < 0:
            self.tokens = 0
        if self.tokens == 0 and self.text:
            self.tokens = max(1, (len(self.text) + 3) // 4)
        return self


class IntentWeights(BaseModel):
    relevance: float = 1.0
    provenance: float = 0.6
    recency: float = 0.5
    freshness_trust: float = 0.4
    novelty: float = 0.3
    intent_fit: float = 0.8
    redundancy_penalty: float = 0.7
    contradiction_penalty: float = 0.8


class ContextSelectionPolicy(BaseModel):
    intent_family: IntentFamily
    discovery_mode: DiscoveryMode = DiscoveryMode.OFF

    hard_reservation_tokens: int = 256
    minimal_fresh_tail: int = 3

    lane_minima: Dict[str, int] = Field(default_factory=dict)
    lane_maxima: Dict[str, int] = Field(default_factory=dict)

    cluster_bonus: float = 0.15
    redundancy_penalty: float = 0.65
    contradiction_penalty: float = 0.8

    front_anchor_budget: int = 2
    back_anchor_budget: int = 2

    per_intent_weights: IntentWeights = Field(default_factory=IntentWeights)


class SelectedBlock(BaseModel):
    id: str
    lane: str
    memory_type: str
    tokens: int
    utility: float
    cluster_id: str | None = None
    provenance: Dict[str, Any] = Field(default_factory=dict)


class DroppedBlock(BaseModel):
    id: str
    lane: str
    tokens: int
    reason: str


class SelectionExplanation(BaseModel):
    intent_family: IntentFamily
    total_budget: int
    reserved_tokens: int
    selected_blocks: List[SelectedBlock] = Field(default_factory=list)
    dropped_blocks: List[DroppedBlock] = Field(default_factory=list)
    lane_totals: Dict[str, int] = Field(default_factory=dict)
    cluster_coverage: List[str] = Field(default_factory=list)
    reorder_strategy: str = "front_back_anchor"
