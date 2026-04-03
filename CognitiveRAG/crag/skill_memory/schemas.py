from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from CognitiveRAG.crag.skill_memory.namespaces import SkillNamespace, namespace_for


ArtifactType = Literal[
    "raw_chunk",
    "principle",
    "template",
    "example",
    "rubric",
    "anti_pattern",
    "workflow",
    "style_gist",
]


class SkillSourceRef(BaseModel):
    source_kind: Literal["craft", "content"] = "craft"
    source_path: str = ""
    book_id: str = ""
    chapter_id: str = ""
    section_id: str = ""
    chunk_id: str = ""
    char_start: int | None = None
    char_end: int | None = None


class SkillArtifact(BaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    namespace: SkillNamespace
    canonical_text: str
    normalized_key: str
    title: str = ""
    confidence: float = 0.6
    tags: List[str] = Field(default_factory=list)
    source_refs: List[SkillSourceRef] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    metadata: Dict[str, object] = Field(default_factory=dict)


class RawChunkArtifact(SkillArtifact):
    artifact_type: Literal["raw_chunk"] = "raw_chunk"


class PrincipleArtifact(SkillArtifact):
    artifact_type: Literal["principle"] = "principle"
    principle: str


class TemplateArtifact(SkillArtifact):
    artifact_type: Literal["template"] = "template"
    template_text: str
    slots: List[str] = Field(default_factory=list)


class ExampleArtifact(SkillArtifact):
    artifact_type: Literal["example"] = "example"
    example_text: str
    verdict: str = ""


class RubricArtifact(SkillArtifact):
    artifact_type: Literal["rubric"] = "rubric"
    criteria: List[str] = Field(default_factory=list)
    scale: str = "unspecified"


class AntiPatternArtifact(SkillArtifact):
    artifact_type: Literal["anti_pattern"] = "anti_pattern"
    anti_pattern: str
    correction_hint: str = ""


class WorkflowArtifact(SkillArtifact):
    artifact_type: Literal["workflow"] = "workflow"
    steps: List[str] = Field(default_factory=list)


class StyleGistArtifact(SkillArtifact):
    artifact_type: Literal["style_gist"] = "style_gist"
    style_traits: List[str] = Field(default_factory=list)


AgentType = Literal["script_agent", "storyboard_agent"]


class SkillPackRequest(BaseModel):
    query: str
    agent_type: AgentType
    task_type: str
    channel_type: str = ""
    language: str = ""
    style_profile: str = ""
    max_items: int = 12


class SkillPack(BaseModel):
    query: str
    agent_type: AgentType
    task_type: str
    channel_type: str = ""
    language: str = ""
    style_profile: str = ""
    selected_artifact_ids: List[str] = Field(default_factory=list)
    grouped_artifacts: Dict[str, List[SkillArtifact]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    selection_explanations: Dict[str, Dict[str, object]] = Field(default_factory=dict)


def normalize_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def build_artifact_id(*, artifact_type: str, namespace: str, normalized_key: str) -> str:
    seed = f"{artifact_type}|{namespace}|{normalized_key}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"skill:{artifact_type}:{digest}"


def build_artifact(
    *,
    artifact_type: ArtifactType,
    source_ref: SkillSourceRef,
    canonical_text: str,
    title: str = "",
    confidence: float = 0.6,
    tags: List[str] | None = None,
    metadata: Dict[str, object] | None = None,
) -> SkillArtifact:
    normalized = normalize_key(canonical_text)
    ns = namespace_for(source_kind=source_ref.source_kind, artifact_type=artifact_type)
    artifact_id = build_artifact_id(artifact_type=artifact_type, namespace=ns, normalized_key=normalized)
    common = dict(
        artifact_id=artifact_id,
        namespace=ns,
        canonical_text=canonical_text.strip(),
        normalized_key=normalized,
        title=title.strip(),
        confidence=float(confidence),
        tags=list(tags or []),
        source_refs=[source_ref],
        metadata=dict(metadata or {}),
    )
    if artifact_type == "raw_chunk":
        return RawChunkArtifact(**common)
    if artifact_type == "principle":
        return PrincipleArtifact(**common, principle=canonical_text.strip())
    if artifact_type == "template":
        slots = []
        for token in canonical_text.replace("{", " {").split():
            if token.startswith("{") and token.endswith("}"):
                slots.append(token[1:-1].strip().lower())
        return TemplateArtifact(**common, template_text=canonical_text.strip(), slots=sorted(set(slots)))
    if artifact_type == "example":
        verdict = "good" if "good" in normalized or "after" in normalized else "neutral"
        return ExampleArtifact(**common, example_text=canonical_text.strip(), verdict=verdict)
    if artifact_type == "rubric":
        criteria = [p.strip() for p in canonical_text.split(";") if p.strip()]
        return RubricArtifact(**common, criteria=criteria, scale="1-5" if "1-5" in canonical_text else "unspecified")
    if artifact_type == "anti_pattern":
        correction = "Avoid this pattern and apply explicit checks."
        return AntiPatternArtifact(**common, anti_pattern=canonical_text.strip(), correction_hint=correction)
    if artifact_type == "workflow":
        parts = [p.strip(" -") for p in canonical_text.replace("→", "->").split("->") if p.strip(" -")]
        return WorkflowArtifact(**common, steps=parts)
    if artifact_type == "style_gist":
        traits = [p.strip() for p in canonical_text.split(",") if p.strip()]
        return StyleGistArtifact(**common, style_traits=traits)
    raise ValueError(f"Unsupported artifact type: {artifact_type}")


def artifact_to_record(artifact: SkillArtifact) -> dict:
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "namespace": artifact.namespace,
        "title": artifact.title,
        "canonical_text": artifact.canonical_text,
        "normalized_key": artifact.normalized_key,
        "confidence": float(artifact.confidence),
        "tags_json": json.dumps(artifact.tags),
        "source_refs_json": json.dumps([ref.model_dump() for ref in artifact.source_refs]),
        "links_json": json.dumps(artifact.links),
        "metadata_json": json.dumps(artifact.metadata),
        "payload_json": json.dumps(artifact.model_dump()),
    }
