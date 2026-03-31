from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionProvenance(BaseModel):
    session_id: str = ""
    run_id: str = ""
    source: str = "skill_execution"
    metadata: Dict[str, object] = Field(default_factory=dict)


class SkillExecutionCase(BaseModel):
    execution_case_id: str
    agent_type: str
    task_type: str
    channel_type: str = ""
    language: str = ""
    request_text: str
    selected_artifact_ids: List[str] = Field(default_factory=list)
    pack_summary: str = ""
    pack_ref: str = ""
    output_text: str = ""
    output_ref: str = ""
    success_flag: bool = False
    human_edits: List[str] = Field(default_factory=list)
    notes: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    provenance: ExecutionProvenance = Field(default_factory=ExecutionProvenance)


def build_execution_case_id(
    *, agent_type: str, task_type: str, channel_type: str, request_text: str, created_at: str
) -> str:
    seed = "|".join([agent_type, task_type, channel_type, request_text.strip().lower(), created_at, str(uuid.uuid4())])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]
    return f"exec:{digest}"


def build_execution_case(
    *,
    agent_type: str,
    task_type: str,
    request_text: str,
    selected_artifact_ids: List[str],
    channel_type: str = "",
    language: str = "",
    pack_summary: str = "",
    pack_ref: str = "",
    output_text: str = "",
    output_ref: str = "",
    success_flag: bool = False,
    human_edits: List[str] | None = None,
    notes: str = "",
    provenance: ExecutionProvenance | None = None,
) -> SkillExecutionCase:
    created = utc_now_iso()
    return SkillExecutionCase(
        execution_case_id=build_execution_case_id(
            agent_type=agent_type,
            task_type=task_type,
            channel_type=channel_type,
            request_text=request_text,
            created_at=created,
        ),
        agent_type=agent_type,
        task_type=task_type,
        channel_type=channel_type,
        language=language,
        request_text=request_text,
        selected_artifact_ids=sorted(set(selected_artifact_ids)),
        pack_summary=pack_summary,
        pack_ref=pack_ref,
        output_text=output_text,
        output_ref=output_ref,
        success_flag=bool(success_flag),
        human_edits=list(human_edits or []),
        notes=notes,
        created_at=created,
        updated_at=created,
        provenance=provenance or ExecutionProvenance(),
    )

