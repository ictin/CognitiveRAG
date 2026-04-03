from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_node_id(node_type: str, key: str) -> str:
    seed = f"{(node_type or '').strip().lower()}::{(key or '').strip()}"
    digest = hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]
    return f"gn:{(node_type or 'unknown').strip().lower()}:{digest}"


def stable_edge_id(source_node_id: str, relation_type: str, target_node_id: str) -> str:
    seed = f"{(source_node_id or '').strip()}|{(relation_type or '').strip().upper()}|{(target_node_id or '').strip()}"
    digest = hashlib.sha1(seed.encode('utf-8')).hexdigest()[:18]
    return f"ge:{digest}"


class GraphRelationType:
    SUPPORTED_BY = 'SUPPORTED_BY'
    DERIVED_FROM = 'DERIVED_FROM'
    RESOLVED_BY = 'RESOLVED_BY'
    BELONGS_TO_TOPIC = 'BELONGS_TO_TOPIC'
    BELONGS_TO_CATEGORY = 'BELONGS_TO_CATEGORY'
    USES_SKILL_ARTIFACT = 'USES_SKILL_ARTIFACT'
    PRODUCED_OUTPUT = 'PRODUCED_OUTPUT'
    EVALUATES_EXECUTION = 'EVALUATES_EXECUTION'
    REINFORCES_SKILL_ARTIFACT = 'REINFORCES_SKILL_ARTIFACT'
    CRITIQUES_SKILL_ARTIFACT = 'CRITIQUES_SKILL_ARTIFACT'


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str | None = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


class GraphEdge(BaseModel):
    edge_id: str
    source_node_id: str
    relation_type: str
    target_node_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
