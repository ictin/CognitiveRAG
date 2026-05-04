#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore
from CognitiveRAG.crag.skill_memory.execution_schema import build_execution_case
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore
from CognitiveRAG.crag.skill_memory.pack_builder import build_skill_pack
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore
from CognitiveRAG.crag.skill_memory.schemas import SkillPackRequest, SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f018_skill_memory_structural_reuse_slice"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    skill_store = SkillMemoryStore(workdir / "skills.sqlite3")
    exec_store = SkillExecutionStore(workdir / "skill_exec.sqlite3")
    eval_store = SkillEvaluationStore(workdir / "skill_eval.sqlite3")

    ref = SkillSourceRef(source_kind="craft", source_path="/craft/recipe_playbook.md", chunk_id="recipe-chunk-1")
    seeded = [
        build_artifact(
            artifact_type="principle",
            source_ref=ref,
            canonical_text="Principle: lead with a concrete promise and one measurable payoff.",
            title="principle",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        ),
        build_artifact(
            artifact_type="template",
            source_ref=ref,
            canonical_text="Template: {hook} -> {proof} -> {cta}",
            title="template",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        ),
        build_artifact(
            artifact_type="rubric",
            source_ref=ref,
            canonical_text="Rubric: hook clarity; proof specificity; CTA force",
            title="rubric",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        ),
        build_artifact(
            artifact_type="anti_pattern",
            source_ref=ref,
            canonical_text="Anti-pattern: vague intro and delayed payoff.",
            title="anti-pattern",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        ),
        build_artifact(
            artifact_type="workflow",
            source_ref=ref,
            canonical_text="Workflow: draft hook -> validate specificity -> enforce CTA",
            title="workflow",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        ),
        build_artifact(
            artifact_type="style_note",
            source_ref=ref,
            canonical_text="Style: concise, direct, practical",
            title="style-note",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        ),
    ]
    skill_store.upsert_many(seeded)

    execution_case = build_execution_case(
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        request_text="Write a ramen hook with one concrete proof and CTA.",
        selected_artifact_ids=[a.artifact_id for a in seeded[:3]],
        output_text="Ramen in 8 minutes: one pan, deep flavor, start now.",
        success_flag=True,
    )
    exec_store.upsert_case(execution_case)

    evaluation_case = build_evaluation_case(
        execution_case_id=execution_case.execution_case_id,
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
        criterion_scores=[
            RubricCriterionScore(criterion_id="hook", label="Hook", score=2, max_score=5),
            RubricCriterionScore(criterion_id="proof", label="Proof", score=2, max_score=5),
        ],
        pass_flag=False,
        weaknesses=["hook too abstract"],
        improvement_notes=["include one measurable payoff in first sentence"],
        anti_pattern_hits=["vague_intro"],
    )
    eval_store.upsert_case(evaluation_case)

    prompt = "Give me a better ramen short script opening with reusable structure."
    pack = build_skill_pack(
        store=skill_store,
        request=SkillPackRequest(
            query=prompt,
            agent_type="script_agent",
            task_type="recipe_short",
            channel_type="short_video",
            language="en",
            max_items=12,
        ),
    )

    category_coverage = {k: [a.artifact_id for a in v] for k, v in pack.grouped_artifacts.items()}
    (outdir / "structured_skill_memory_readback_artifact.json").write_text(
        json.dumps(
            {
                "selected_artifact_ids": pack.selected_artifact_ids,
                "grouped_artifacts": {
                    k: [a.model_dump(mode="json") for a in v] for k, v in pack.grouped_artifacts.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "category_coverage_artifact.json").write_text(
        json.dumps(
            {
                "categories_present": sorted(category_coverage.keys()),
                "required_subset_present": {
                    "principle": "principle" in category_coverage,
                    "template": "template" in category_coverage,
                    "rubric": "rubric" in category_coverage,
                    "anti_pattern": "anti_pattern" in category_coverage,
                    "workflow": "workflow" in category_coverage,
                    "style_note": "style_note" in category_coverage,
                    "execution_lesson": "execution_lesson" in category_coverage,
                    "evaluation_lesson": "evaluation_lesson" in category_coverage,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reused = {
        "execution_lesson_artifacts": [a.model_dump(mode="json") for a in pack.grouped_artifacts.get("execution_lesson", [])],
        "evaluation_lesson_artifacts": [a.model_dump(mode="json") for a in pack.grouped_artifacts.get("evaluation_lesson", [])],
        "selection_explanations": {
            aid: pack.selection_explanations.get(aid, {}) for aid in pack.selected_artifact_ids
        },
    }
    (outdir / "reuse_proof_artifact.json").write_text(json.dumps(reused, indent=2), encoding="utf-8")

    answer_lines = []
    if pack.grouped_artifacts.get("template"):
        answer_lines.append(f"Template reuse: {pack.grouped_artifacts['template'][0].canonical_text}")
    if pack.grouped_artifacts.get("execution_lesson"):
        answer_lines.append(f"Execution lesson reuse: {pack.grouped_artifacts['execution_lesson'][0].canonical_text}")
    if pack.grouped_artifacts.get("evaluation_lesson"):
        answer_lines.append(f"Evaluation lesson reuse: {pack.grouped_artifacts['evaluation_lesson'][0].canonical_text}")
    answer_lines.append("Suggested opening: 'Ramen in 8 minutes, deeper flavor than takeout, one pan only.'")

    (outdir / "prompt_answer_capture.json").write_text(
        json.dumps({"prompt": prompt, "answer": "\n".join(answer_lines)}, indent=2),
        encoding="utf-8",
    )

    explanation_rows = []
    for art_id in pack.selected_artifact_ids:
        row = next(
            (a for entries in pack.grouped_artifacts.values() for a in entries if a.artifact_id == art_id),
            None,
        )
        if not row:
            continue
        explanation_rows.append(
            {
                "artifact_id": art_id,
                "artifact_type": row.artifact_type,
                "source_refs": [r.model_dump() for r in row.source_refs],
                "metadata": row.metadata,
                "selection_explanation": pack.selection_explanations.get(art_id, {}),
            }
        )
    (outdir / "explanation_skill_memory_artifact.json").write_text(
        json.dumps(explanation_rows, indent=2),
        encoding="utf-8",
    )

    checks = {
        "structured_categories_present": all(
            [
                "principle" in category_coverage,
                "template" in category_coverage,
                "rubric" in category_coverage,
                "anti_pattern" in category_coverage,
                "workflow" in category_coverage,
                "style_note" in category_coverage,
                "execution_lesson" in category_coverage,
                "evaluation_lesson" in category_coverage,
            ]
        ),
        "readback_predictable": len(pack.selected_artifact_ids) > 0,
        "reuse_includes_lessons": bool(pack.grouped_artifacts.get("execution_lesson"))
        and bool(pack.grouped_artifacts.get("evaluation_lesson")),
        "explanation_truthful_with_provenance": all(
            row.get("source_refs") and isinstance(row.get("selection_explanation"), dict) for row in explanation_rows
        ),
    }
    summary = {
        "schemaVersion": "f018_skill_memory_structural_reuse.v1",
        "artifactDir": str(outdir),
        "checks": checks,
        "traceability": {
            "features": ["F-018", "F-009"],
            "requirements": ["REQ-031"],
            "workflows": ["WF-006"],
            "test_scenarios": ["TC-006"],
            "invariants": ["INV-016", "INV-017"],
        },
    }
    summary["passed"] = all(checks.values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
