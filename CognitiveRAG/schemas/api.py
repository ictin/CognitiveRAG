from pydantic import BaseModel, Field

from CognitiveRAG.schemas.agent import OrchestrationTrace
from CognitiveRAG.schemas.memory import EpisodicEvent, ProfileFact, ReasoningPattern, TaskRecord


class QueryRequest(BaseModel):
    query: str
    project: str | None = None
    task_id: str | None = None
    lexical_only: bool = False
    retrieval_mode: str | None = None


class QueryResponse(BaseModel):
    answer: str
    trace: OrchestrationTrace | None = None


class IngestPathRequest(BaseModel):
    path: str
    recursive: bool = True


class IngestResponse(BaseModel):
    accepted: bool
    message: str
    document_ids: list[str] = Field(default_factory=list)


class UpsertEventRequest(BaseModel):
    event: EpisodicEvent


class UpsertProfileFactRequest(BaseModel):
    fact: ProfileFact


class UpsertTaskRequest(BaseModel):
    task: TaskRecord


class UpsertReasoningRequest(BaseModel):
    pattern: ReasoningPattern
