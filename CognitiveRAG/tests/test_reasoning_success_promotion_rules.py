import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve as retrieve_promoted
from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore
from CognitiveRAG.crag.skill_memory.execution_schema import build_execution_case
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore
from CognitiveRAG.memory.reasoning_store import ReasoningStore
from CognitiveRAG.memory.reasoning_success import refresh_reasoning_success_signals
from CognitiveRAG.schemas.memory import ReasoningPattern


def _seed_reasoning(tmp_path: Path, *, pattern_id: str = "rp_success", confidence: float = 0.45) -> Path:
    db = tmp_path / "reasoning.sqlite3"
    store = ReasoningStore(db)
    store.upsert(
        ReasoningPattern(
            pattern_id=pattern_id,
            problem_signature="debug timeout runtime churn",
            reasoning_steps=["diagnose", "bound", "verify"],
            solution_summary="Use bounded diagnosis and restart only when required.",
            confidence=confidence,
            provenance=["src://seed"],
        )
    )
    return db


def _seed_exec_eval_cases(tmp_path: Path) -> tuple[SkillExecutionStore, SkillEvaluationStore]:
    skill_dir = tmp_path / "skill_memory"
    skill_dir.mkdir(parents=True, exist_ok=True)
    exec_store = SkillExecutionStore(skill_dir / "skill_execution.sqlite3")
    eval_store = SkillEvaluationStore(skill_dir / "skill_evaluation.sqlite3")
    return exec_store, eval_store


def _pass_scores() -> list[RubricCriterionScore]:
    return [
        RubricCriterionScore(criterion_id="clarity", label="Clarity", score=4.5, max_score=5.0, weight=1.0),
        RubricCriterionScore(criterion_id="correctness", label="Correctness", score=4.8, max_score=5.0, weight=1.0),
    ]


def _fail_scores() -> list[RubricCriterionScore]:
    return [
        RubricCriterionScore(criterion_id="clarity", label="Clarity", score=1.4, max_score=5.0, weight=1.0),
        RubricCriterionScore(criterion_id="correctness", label="Correctness", score=1.2, max_score=5.0, weight=1.0),
    ]


def _row(db_path: Path, pattern_id: str):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT confidence, success_signal_count, failure_signal_count, success_confidence, success_basis_json "
            "FROM reasoning_patterns WHERE pattern_id=?",
            (pattern_id,),
        ).fetchone()


def test_repeated_success_accumulates_confidence_deterministically(tmp_path: Path):
    db_path = _seed_reasoning(tmp_path, pattern_id="rp_a", confidence=0.40)
    exec_store, eval_store = _seed_exec_eval_cases(tmp_path)

    exec_one = build_execution_case(
        agent_type="script_agent",
        task_type="short_recipe",
        request_text="Generate script 1",
        selected_artifact_ids=["rp_a"],
        success_flag=True,
        output_text="ok",
    )
    exec_two = build_execution_case(
        agent_type="script_agent",
        task_type="short_recipe",
        request_text="Generate script 2",
        selected_artifact_ids=["rp_a"],
        success_flag=True,
        output_text="ok",
    )
    exec_store.upsert_case(exec_one)
    exec_store.upsert_case(exec_two)
    eval_store.upsert_case(
        build_evaluation_case(
            execution_case_id=exec_one.execution_case_id,
            agent_type="script_agent",
            task_type="short_recipe",
            criterion_scores=_pass_scores(),
            pass_flag=True,
            rubric_id="rubric:script",
        )
    )
    eval_store.upsert_case(
        build_evaluation_case(
            execution_case_id=exec_two.execution_case_id,
            agent_type="script_agent",
            task_type="short_recipe",
            criterion_scores=_pass_scores(),
            pass_flag=True,
            rubric_id="rubric:script",
        )
    )

    first = refresh_reasoning_success_signals(workdir=tmp_path, reasoning_db_path=db_path)
    second = refresh_reasoning_success_signals(workdir=tmp_path, reasoning_db_path=db_path)
    assert first["updated_patterns"] >= 1
    assert second["updated_patterns"] >= 1

    row = _row(db_path, "rp_a")
    assert row is not None
    assert int(row["success_signal_count"]) == 2
    assert int(row["failure_signal_count"]) == 0
    assert float(row["success_confidence"]) > float(row["confidence"])
    # Running refresh repeatedly with unchanged inputs should remain stable.
    stable_value = float(row["success_confidence"])
    refresh_reasoning_success_signals(workdir=tmp_path, reasoning_db_path=db_path)
    row_after = _row(db_path, "rp_a")
    assert round(float(row_after["success_confidence"]), 6) == round(stable_value, 6)


def test_growth_is_capped_and_failures_do_not_inflate(tmp_path: Path):
    db_path = _seed_reasoning(tmp_path, pattern_id="rp_b", confidence=0.35)
    exec_store, eval_store = _seed_exec_eval_cases(tmp_path)

    # Many successes should still cap confidence <= 1.
    for i in range(20):
        case = build_execution_case(
            agent_type="script_agent",
            task_type="short_recipe",
            request_text=f"run success {i}",
            selected_artifact_ids=["rp_b"],
            success_flag=True,
            output_text="ok",
        )
        exec_store.upsert_case(case)
        eval_store.upsert_case(
            build_evaluation_case(
                execution_case_id=case.execution_case_id,
                agent_type="script_agent",
                task_type="short_recipe",
                criterion_scores=_pass_scores(),
                pass_flag=True,
                rubric_id="rubric:script",
            )
        )

    # Failures should reduce confidence gain.
    for i in range(3):
        case = build_execution_case(
            agent_type="script_agent",
            task_type="short_recipe",
            request_text=f"run fail {i}",
            selected_artifact_ids=["rp_b"],
            success_flag=False,
            output_text="bad",
        )
        exec_store.upsert_case(case)
        eval_store.upsert_case(
            build_evaluation_case(
                execution_case_id=case.execution_case_id,
                agent_type="script_agent",
                task_type="short_recipe",
                criterion_scores=_fail_scores(),
                pass_flag=False,
                rubric_id="rubric:script",
            )
        )

    refresh_reasoning_success_signals(workdir=tmp_path, reasoning_db_path=db_path)
    row = _row(db_path, "rp_b")
    assert row is not None
    assert int(row["success_signal_count"]) == 20
    assert int(row["failure_signal_count"]) == 3
    assert 0.0 <= float(row["success_confidence"]) <= 1.0


def test_missing_or_unlinked_outcomes_are_not_treated_as_success(tmp_path: Path):
    db_path = _seed_reasoning(tmp_path, pattern_id="rp_c", confidence=0.60)
    exec_store, eval_store = _seed_exec_eval_cases(tmp_path)

    # Execution case without selected artifact linkage should not create success for rp_c.
    case = build_execution_case(
        agent_type="script_agent",
        task_type="short_recipe",
        request_text="unlinked",
        selected_artifact_ids=[],
        success_flag=True,
        output_text="ok",
    )
    exec_store.upsert_case(case)
    eval_store.upsert_case(
        build_evaluation_case(
            execution_case_id=case.execution_case_id,
            agent_type="script_agent",
            task_type="short_recipe",
            criterion_scores=_pass_scores(),
            pass_flag=True,
            rubric_id="rubric:script",
        )
    )

    refresh_reasoning_success_signals(workdir=tmp_path, reasoning_db_path=db_path)
    row = _row(db_path, "rp_c")
    assert row is not None
    assert int(row["success_signal_count"]) == 0
    assert float(row["success_confidence"]) <= float(row["confidence"])


def test_promoted_lane_applies_bounded_success_confidence_helper(tmp_path: Path):
    db_path = tmp_path / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT, "
            "exact_fingerprint TEXT, near_fingerprint TEXT, canonical_pattern_id TEXT, near_duplicate_of TEXT, reuse_count INTEGER, merged_from_json TEXT, "
            "success_signal_count INTEGER, failure_signal_count INTEGER, success_confidence REAL, success_basis_json TEXT)"
        )
        conn.execute(
            "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "rp_low_success",
                "sig",
                "[]",
                "bounded diagnosis fallback",
                0.7,
                "[]",
                "workflow_pattern",
                "bounded diagnosis fallback",
                "current",
                None,
                None,
                "rp_low_success",
                None,
                1,
                "[]",
                0,
                0,
                0.0,
                "{}",
            ),
        )
        conn.execute(
            "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "rp_high_success",
                "sig",
                "[]",
                "bounded diagnosis fallback",
                0.7,
                "[]",
                "workflow_pattern",
                "bounded diagnosis fallback",
                "current",
                None,
                None,
                "rp_high_success",
                None,
                1,
                "[]",
                5,
                0,
                1.0,
                "{\"source\":\"test\"}",
            ),
        )
        conn.commit()

    hits = retrieve_promoted(
        workdir=str(tmp_path),
        intent_family=IntentFamily.INVESTIGATIVE,
        query="bounded diagnosis",
        top_k=2,
    )
    assert [h.id for h in hits] == ["promoted:rp_high_success", "promoted:rp_low_success"]
    delta = hits[0].semantic_score - hits[1].semantic_score
    assert 0.0 < delta <= 0.12
    assert hits[0].provenance.get("success_confidence") == 1.0
    assert hits[1].provenance.get("success_confidence") == 0.0

