from CognitiveRAG.crag.skill_memory.ranking import rank_artifacts
from CognitiveRAG.crag.skill_memory.schemas import SkillPackRequest, SkillSourceRef, build_artifact


def _artifact(artifact_type: str, text: str, confidence: float = 0.8, tags: list[str] | None = None):
    return build_artifact(
        artifact_type=artifact_type,  # type: ignore[arg-type]
        source_ref=SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id=f"{artifact_type}-{text[:8]}"),
        canonical_text=text,
        confidence=confidence,
        tags=tags or [],
        metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
    )


def test_type_priority_ranks_typed_above_raw_chunks():
    request = SkillPackRequest(
        query="hook and pacing",
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
    )
    ranked = rank_artifacts(
        [
            _artifact("raw_chunk", "Raw chunk: hook then pacing tips", 0.95),
            _artifact("principle", "Principle: hook in first five seconds", 0.8),
            _artifact("template", "Template: {hook} then {benefit}", 0.75),
        ],
        request,
    )
    assert ranked[0].artifact.artifact_type == "principle"
    assert ranked[1].artifact.artifact_type == "template"
    assert ranked[-1].artifact.artifact_type == "raw_chunk"

