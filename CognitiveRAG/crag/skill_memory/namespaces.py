from __future__ import annotations

from typing import Literal


SkillNamespace = Literal[
    "craft_raw",
    "craft_principles",
    "craft_templates",
    "craft_examples",
    "craft_rubrics",
    "craft_antipatterns",
    "craft_workflows",
    "style_profiles",
    "content_books_raw",
]


def namespace_for(*, source_kind: str, artifact_type: str) -> SkillNamespace:
    sk = str(source_kind or "").strip().lower()
    at = str(artifact_type or "").strip().lower()
    if sk == "content":
        return "content_books_raw"
    if at == "raw_chunk":
        return "craft_raw"
    if at == "principle":
        return "craft_principles"
    if at == "template":
        return "craft_templates"
    if at == "example":
        return "craft_examples"
    if at == "rubric":
        return "craft_rubrics"
    if at == "anti_pattern":
        return "craft_antipatterns"
    if at == "workflow":
        return "craft_workflows"
    if at == "style_gist":
        return "style_profiles"
    return "craft_raw"

