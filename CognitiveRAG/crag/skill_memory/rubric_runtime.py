from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class RubricCriterionScore(BaseModel):
    criterion_id: str
    label: str
    score: float
    max_score: float = 5.0
    weight: float = 1.0
    notes: str = ""


class RubricRuntime(BaseModel):
    rubric_id: str
    rubric_ref: str = ""
    criteria: List[RubricCriterionScore] = Field(default_factory=list)


def compute_weighted_score(criteria: List[RubricCriterionScore]) -> float:
    if not criteria:
        return 0.0
    weighted_sum = 0.0
    weighted_max = 0.0
    for c in criteria:
        weight = max(0.0, float(c.weight))
        max_score = max(1e-9, float(c.max_score))
        score = min(max(float(c.score), 0.0), max_score)
        weighted_sum += score * weight
        weighted_max += max_score * weight
    if weighted_max <= 0:
        return 0.0
    return weighted_sum / weighted_max

