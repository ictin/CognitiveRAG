from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    objective: str
    steps: list[str] = Field(default_factory=list)


class CriticOutput(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)


class EntityRelationExtraction(BaseModel):
    entities: list[str] = Field(default_factory=list)
    relations: list[dict] = Field(default_factory=list)
