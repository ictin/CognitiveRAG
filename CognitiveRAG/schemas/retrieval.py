from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str | None = None
    text: str
    source_type: str
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)
    # ranking/debug fields
    rank: int | None = None
    final_score: float | None = None
    ranking_reason: str | None = None


class RetrievalBundle(BaseModel):
    query: str
    intent: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    # minimal augmentation decision produced by retriever
    augmentation_decision: dict | None = None
