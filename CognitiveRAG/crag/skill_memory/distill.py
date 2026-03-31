from __future__ import annotations

import re
from typing import List

from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact, SkillSourceRef, build_artifact


_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")


def _sentences(text: str) -> List[str]:
    chunks = [s.strip() for s in _SPLIT_RE.split(str(text or ""))]
    return [s for s in chunks if s]


def _is_workflow(s: str) -> bool:
    l = s.lower()
    return "->" in s or "→" in s or "step " in l or "then " in l or "workflow" in l


def _is_rubric(s: str) -> bool:
    l = s.lower()
    return "rubric" in l or "score" in l or "criteria" in l or "1-5" in l


def _is_template(s: str) -> bool:
    l = s.lower()
    return "template" in l or ("{" in s and "}" in s) or "[hook]" in l


def _is_example(s: str) -> bool:
    l = s.lower()
    return "before:" in l or "after:" in l or "good:" in l or "bad:" in l or "example" in l


def _is_antipattern(s: str) -> bool:
    l = s.lower()
    return "anti-pattern" in l or "mistake" in l or "avoid" in l or "don't" in l


def _is_style_gist(s: str) -> bool:
    l = s.lower()
    return "tone" in l or "voice" in l or "style" in l


def _is_principle(s: str) -> bool:
    l = s.lower()
    return "principle" in l or "rule" in l or "should" in l or "always" in l or "never" in l


def distill_chunk(
    *,
    text: str,
    source_ref: SkillSourceRef,
    include_raw: bool = True,
) -> List[SkillArtifact]:
    artifacts: List[SkillArtifact] = []
    body = str(text or "").strip()
    if not body:
        return artifacts

    if include_raw:
        artifacts.append(
            build_artifact(
                artifact_type="raw_chunk",
                source_ref=source_ref,
                canonical_text=body[:6000],
                title="raw_chunk",
                confidence=1.0,
            )
        )

    if source_ref.source_kind == "content":
        return artifacts

    for sentence in _sentences(body):
        art_type = None
        if _is_rubric(sentence):
            art_type = "rubric"
        elif _is_template(sentence):
            art_type = "template"
        elif _is_example(sentence):
            art_type = "example"
        elif _is_antipattern(sentence):
            art_type = "anti_pattern"
        elif _is_workflow(sentence):
            art_type = "workflow"
        elif _is_style_gist(sentence):
            art_type = "style_gist"
        elif _is_principle(sentence):
            art_type = "principle"
        if art_type is None:
            continue
        artifacts.append(
            build_artifact(
                artifact_type=art_type,  # type: ignore[arg-type]
                source_ref=source_ref,
                canonical_text=sentence[:2000],
                title=art_type,
                confidence=0.75,
            )
        )
    return artifacts
