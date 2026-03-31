from pathlib import Path

from CognitiveRAG.crag.skill_memory.pack_builder import build_skill_pack
from CognitiveRAG.crag.skill_memory.schemas import SkillPackRequest, SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


def _seed_script_artifacts(store: SkillMemoryStore) -> None:
    ref = SkillSourceRef(source_kind="craft", source_path="/script_craft.md", chunk_id="script-1")
    store.upsert(
        build_artifact(
            artifact_type="principle",
            source_ref=ref,
            canonical_text="Principle: Open with a concrete hook in 5 seconds.",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short", "channel_type": "short_video"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="template",
            source_ref=ref,
            canonical_text="Template: {hook} then {pain} then {payoff}.",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="example",
            source_ref=ref,
            canonical_text="Example: Before: vague intro. After: direct promise and visual setup.",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="rubric",
            source_ref=ref,
            canonical_text="Rubric: hook clarity; pacing; payoff; CTA",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="anti_pattern",
            source_ref=ref,
            canonical_text="Anti-pattern: slow intro with no promise.",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        )
    )


def _seed_storyboard_artifacts(store: SkillMemoryStore) -> None:
    ref = SkillSourceRef(source_kind="craft", source_path="/storyboard_craft.md", chunk_id="story-1")
    store.upsert(
        build_artifact(
            artifact_type="principle",
            source_ref=ref,
            canonical_text="Principle: each scene needs one visual intent.",
            metadata={"agent_type": "storyboard_agent", "task_type": "short_explainer", "channel_type": "youtube"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="template",
            source_ref=ref,
            canonical_text="Template: Scene {setup} -> Scene {turn} -> Scene {resolution}.",
            metadata={"agent_type": "storyboard_agent", "task_type": "short_explainer"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="example",
            source_ref=ref,
            canonical_text="Example: Good: cut on action to keep pacing.",
            metadata={"agent_type": "storyboard_agent", "task_type": "short_explainer"},
        )
    )


def test_script_agent_pack_is_bounded_and_prioritized(tmp_path: Path):
    store = SkillMemoryStore(tmp_path / "skills.sqlite3")
    _seed_script_artifacts(store)

    pack = build_skill_pack(
        store=store,
        request=SkillPackRequest(
            query="recipe short pacing and hooks",
            agent_type="script_agent",
            task_type="recipe_short",
            channel_type="short_video",
            language="en",
            max_items=10,
        ),
    )
    assert pack.selected_artifact_ids
    assert "principle" in pack.grouped_artifacts
    assert "template" in pack.grouped_artifacts
    assert "example" in pack.grouped_artifacts
    assert len(pack.grouped_artifacts["principle"]) <= 5
    assert len(pack.grouped_artifacts.get("template", [])) <= 2
    assert len(pack.grouped_artifacts.get("example", [])) <= 3
    assert len(pack.selected_artifact_ids) <= 10


def test_storyboard_agent_pack_prefers_story_structure_artifacts(tmp_path: Path):
    store = SkillMemoryStore(tmp_path / "skills.sqlite3")
    _seed_storyboard_artifacts(store)

    pack = build_skill_pack(
        store=store,
        request=SkillPackRequest(
            query="scene pacing for explainer",
            agent_type="storyboard_agent",
            task_type="short_explainer",
            channel_type="youtube",
            language="en",
            max_items=8,
        ),
    )
    assert pack.grouped_artifacts.get("principle")
    assert pack.grouped_artifacts.get("template")
    assert pack.grouped_artifacts.get("example")

