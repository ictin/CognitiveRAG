from datetime import datetime, timezone
import hashlib
from pydantic import BaseModel, Field
from pydantic import model_validator


class MemoryItem(BaseModel):
    item_id: str
    item_type: str
    project: str | None = None
    task_id: str | None = None
    entity_id: str | None = None
    source: str = "unknown"
    provenance: dict = Field(default_factory=dict)
    summarizable: bool = True
    exactness: str = "derived"
    content: str = ""
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def can_summarize(self) -> bool:
        return self.summarizable and self.exactness != "exact"

    def is_exact(self) -> bool:
        return self.exactness == "exact"


class ArtifactExact(MemoryItem):
    item_type: str = "artifact_exact"
    summarizable: bool = False
    exactness: str = "exact"


class FactPinned(MemoryItem):
    item_type: str = "fact_pinned"
    summarizable: bool = False
    exactness: str = "exact"


class EpisodicEvent(MemoryItem):
    item_type: str = "episodic_event"
    summarizable: bool = True
    exactness: str = "derived"
    event_type: str | None = None
    result: str | None = None


class TaskState(MemoryItem):
    item_type: str = "task_state"
    summarizable: bool = True
    exactness: str = "derived"
    status: str = "unknown"


class ProjectState(MemoryItem):
    item_type: str = "project_state"
    summarizable: bool = True
    exactness: str = "derived"
    status: str = "unknown"


class DerivedSummary(MemoryItem):
    item_type: str = "derived_summary"
    summarizable: bool = False
    exactness: str = "derived"


class ProfileFact(MemoryItem):
    item_type: str = "profile_fact"
    summarizable: bool = False
    exactness: str = "exact"
    key: str | None = None
    value: str | None = None
    source: str = "user"
    confidence: float = 1.0


class TaskRecord(MemoryItem):
    item_type: str = "task_record"
    summarizable: bool = True
    exactness: str = "derived"
    title: str = ""
    status: str = "unknown"
    summary: str = ""


class ReasoningPattern(MemoryItem):
    item_type: str = "reasoning_pattern"
    summarizable: bool = True
    exactness: str = "derived"
    item_id: str = ""
    pattern_id: str | None = None
    problem_signature: str | None = None
    reasoning_steps: list[str] = Field(default_factory=list)
    solution_summary: str = ""
    confidence: float = 0.0
    provenance: list[str] = Field(default_factory=list)
    memory_subtype: str = "generic"
    normalized_text: str | None = None
    freshness_state: str = "unknown"

    @model_validator(mode="after")
    def _ensure_ids(self):
        pattern_id = (self.pattern_id or "").strip()
        item_id = (self.item_id or "").strip()
        if not pattern_id and item_id:
            pattern_id = item_id
        if not item_id and pattern_id:
            item_id = pattern_id
        if not pattern_id:
            seed = f"{self.problem_signature or ''}|{self.solution_summary or ''}|{self.memory_subtype or ''}"
            pattern_id = f"rp:{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"
        if not item_id:
            item_id = pattern_id
        self.pattern_id = pattern_id
        self.item_id = item_id
        if not self.normalized_text:
            self.normalized_text = self.solution_summary.strip().lower()
        return self


class MemoryContextBlock(BaseModel):
    block_id: str
    session_id: str | None = None
    project: str | None = None
    task_id: str | None = None
    exact_items: list[MemoryItem] = Field(default_factory=list)
    derived_items: list[MemoryItem] = Field(default_factory=list)
    exactness: str = "derived"
    summarizable: bool = True
    provenance: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_prompt_payload(self) -> dict:
        return {
            "block_id": self.block_id,
            "session_id": self.session_id,
            "project": self.project,
            "task_id": self.task_id,
            "summarizable": self.summarizable,
            "exactness": self.exactness,
            "provenance": self.provenance,
            "exact_items": [item.model_dump() for item in self.exact_items],
            "derived_items": [item.model_dump() for item in self.derived_items],
        }


def build_context_block(
    block_id: str,
    *,
    session_id: str | None = None,
    project: str | None = None,
    task_id: str | None = None,
    exact_items: list[MemoryItem] | None = None,
    derived_items: list[MemoryItem] | None = None,
    provenance: dict | None = None,
) -> MemoryContextBlock:
    exact_items = exact_items or []
    derived_items = derived_items or []
    return MemoryContextBlock(
        block_id=block_id,
        session_id=session_id,
        project=project,
        task_id=task_id,
        exact_items=exact_items,
        derived_items=derived_items,
        exactness="exact" if exact_items and not derived_items else "derived",
        summarizable=bool(derived_items),
        provenance=provenance or {},
    )
