from pathlib import Path

from CognitiveRAG.crag.skill_memory.pack_builder import build_skill_pack
from CognitiveRAG.crag.skill_memory.schemas import SkillPackRequest, SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


def test_agent_task_channel_language_hints_affect_ranking(tmp_path: Path):
    store = SkillMemoryStore(tmp_path / "skills.sqlite3")
    ref = SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="match-1")

    matching = build_artifact(
        artifact_type="principle",
        source_ref=ref,
        canonical_text="Principle: storyboard scenes should land one visual beat each.",
        confidence=0.8,
        metadata={
            "agent_type": "storyboard_agent",
            "task_type": "short_explainer",
            "channel_type": "youtube",
            "language": "en",
        },
    )
    non_matching = build_artifact(
        artifact_type="principle",
        source_ref=ref,
        canonical_text="Principle: hooks for short recipe scripts.",
        confidence=0.9,
        metadata={
            "agent_type": "script_agent",
            "task_type": "recipe_short",
            "channel_type": "short_video",
            "language": "es",
        },
    )
    store.upsert(matching)
    store.upsert(non_matching)

    pack = build_skill_pack(
        store=store,
        request=SkillPackRequest(
            query="storyboard explainer scene pacing",
            agent_type="storyboard_agent",
            task_type="short_explainer",
            channel_type="youtube",
            language="en",
            max_items=5,
        ),
    )

    selected_principles = pack.grouped_artifacts.get("principle", [])
    assert selected_principles
    assert selected_principles[0].artifact_id == matching.artifact_id

