from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: str
    description: str
    tool_hint: str | None = None


class Plan(BaseModel):
    objective: str
    steps: list[PlanStep] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class Critique(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    follow_up_actions: list[str] = Field(default_factory=list)


class OrchestrationTrace(BaseModel):
    plan: Plan
    critique: Critique | None = None
    retrieval_summary: list[str] = Field(default_factory=list)
    retrieval_sources: list[dict] = Field(default_factory=list)
    augmentation_decision: dict | None = None
