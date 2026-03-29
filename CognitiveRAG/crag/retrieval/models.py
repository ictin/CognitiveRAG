from __future__ import annotations

from pydantic import BaseModel, Field

from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.types import estimate_tokens


class LaneHit(BaseModel):
    id: str
    lane: RetrievalLane
    memory_type: MemoryType
    text: str
    provenance: dict = Field(default_factory=dict)

    tokens: int = 0
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

    def with_token_estimate(self) -> "LaneHit":
        if self.tokens <= 0:
            self.tokens = estimate_tokens(self.text)
        return self
