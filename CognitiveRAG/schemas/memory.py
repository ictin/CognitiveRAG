from datetime import datetime
from pydantic import BaseModel, Field


class EpisodicEvent(BaseModel):
    event_id: str
    timestamp: datetime
    event_type: str
    goal: str
    result: str | None = None
    success_score: float | None = None
    metadata: dict = Field(default_factory=dict)


class ProfileFact(BaseModel):
    key: str
    value: str
    source: str = "user"
    confidence: float = 1.0


class TaskRecord(BaseModel):
    task_id: str
    title: str
    status: str
    summary: str = ""
    metadata: dict = Field(default_factory=dict)


class ReasoningPattern(BaseModel):
    pattern_id: str
    problem_signature: str
    reasoning_steps: list[str] = Field(default_factory=list)
    solution_summary: str
    confidence: float = 0.0
    provenance: list[str] = Field(default_factory=list)
