from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

from CognitiveRAG.crag.skill_memory.evaluation_linker import evaluation_link_row, has_valid_execution_link
from CognitiveRAG.crag.skill_memory.evaluation_schema import SkillEvaluationCase


class SkillEvaluationStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_evaluation_cases (
                    evaluation_case_id TEXT PRIMARY KEY,
                    execution_case_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    channel_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    rubric_id TEXT NOT NULL,
                    rubric_ref TEXT NOT NULL,
                    criterion_scores_json TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    pass_flag INTEGER NOT NULL,
                    anti_pattern_hits_json TEXT NOT NULL,
                    strengths_json TEXT NOT NULL,
                    weaknesses_json TEXT NOT NULL,
                    human_edits_summary TEXT NOT NULL,
                    improvement_notes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_evaluation_execution_links (
                    evaluation_case_id TEXT PRIMARY KEY,
                    execution_case_id TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_agent_task ON skill_evaluation_cases(agent_type, task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_channel_lang ON skill_evaluation_cases(channel_type, language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_score ON skill_evaluation_cases(overall_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_pass ON skill_evaluation_cases(pass_flag)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_exec_link ON skill_evaluation_execution_links(execution_case_id)"
            )

    def upsert_case(self, case: SkillEvaluationCase) -> None:
        if not has_valid_execution_link(case):
            raise ValueError("evaluation case must include a valid execution_case_id")
        payload = case.model_dump()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skill_evaluation_cases(
                    evaluation_case_id, execution_case_id, agent_type, task_type, channel_type, language,
                    rubric_id, rubric_ref, criterion_scores_json, overall_score, pass_flag, anti_pattern_hits_json,
                    strengths_json, weaknesses_json, human_edits_summary, improvement_notes_json,
                    created_at, updated_at, provenance_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_case_id) DO UPDATE SET
                    execution_case_id=excluded.execution_case_id,
                    agent_type=excluded.agent_type,
                    task_type=excluded.task_type,
                    channel_type=excluded.channel_type,
                    language=excluded.language,
                    rubric_id=excluded.rubric_id,
                    rubric_ref=excluded.rubric_ref,
                    criterion_scores_json=excluded.criterion_scores_json,
                    overall_score=excluded.overall_score,
                    pass_flag=excluded.pass_flag,
                    anti_pattern_hits_json=excluded.anti_pattern_hits_json,
                    strengths_json=excluded.strengths_json,
                    weaknesses_json=excluded.weaknesses_json,
                    human_edits_summary=excluded.human_edits_summary,
                    improvement_notes_json=excluded.improvement_notes_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    provenance_json=excluded.provenance_json,
                    payload_json=excluded.payload_json
                """,
                (
                    case.evaluation_case_id,
                    case.execution_case_id,
                    case.agent_type,
                    case.task_type,
                    case.channel_type,
                    case.language,
                    case.rubric_id,
                    case.rubric_ref,
                    json.dumps([c.model_dump() for c in case.criterion_scores]),
                    float(case.overall_score),
                    int(case.pass_flag),
                    json.dumps(case.anti_pattern_hits),
                    json.dumps(case.strengths),
                    json.dumps(case.weaknesses),
                    case.human_edits_summary,
                    json.dumps(case.improvement_notes),
                    case.created_at,
                    case.updated_at,
                    json.dumps(case.provenance.model_dump()),
                    json.dumps(payload),
                ),
            )
            conn.execute(
                """
                INSERT INTO skill_evaluation_execution_links(evaluation_case_id, execution_case_id)
                VALUES (?, ?)
                ON CONFLICT(evaluation_case_id) DO UPDATE SET
                    execution_case_id=excluded.execution_case_id
                """,
                evaluation_link_row(case),
            )

    def upsert_many(self, cases: Iterable[SkillEvaluationCase]) -> int:
        count = 0
        for case in cases:
            self.upsert_case(case)
            count += 1
        return count

    def get_case(self, evaluation_case_id: str) -> SkillEvaluationCase | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM skill_evaluation_cases WHERE evaluation_case_id = ?",
                (evaluation_case_id,),
            ).fetchone()
        if not row:
            return None
        return SkillEvaluationCase.model_validate(json.loads(row["payload_json"]))

    def list_cases(
        self,
        *,
        agent_type: str | None = None,
        task_type: str | None = None,
        channel_type: str | None = None,
        language: str | None = None,
        pass_flag: bool | None = None,
        limit: int = 30,
    ) -> List[SkillEvaluationCase]:
        where = []
        params: List[object] = []
        if agent_type:
            where.append("agent_type = ?")
            params.append(agent_type)
        if task_type:
            where.append("task_type = ?")
            params.append(task_type)
        if channel_type:
            where.append("channel_type = ?")
            params.append(channel_type)
        if language:
            where.append("language = ?")
            params.append(language)
        if pass_flag is not None:
            where.append("pass_flag = ?")
            params.append(int(pass_flag))
        where_sql = " AND ".join(where) if where else "1=1"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload_json FROM skill_evaluation_cases
                WHERE {where_sql}
                ORDER BY overall_score DESC, created_at DESC
                LIMIT ?
                """,
                (*params, int(limit)),
            ).fetchall()
        return [SkillEvaluationCase.model_validate(json.loads(r["payload_json"])) for r in rows]

    def list_by_execution_case(self, execution_case_id: str, *, limit: int = 20) -> List[SkillEvaluationCase]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.payload_json
                FROM skill_evaluation_cases e
                JOIN skill_evaluation_execution_links l
                  ON l.evaluation_case_id = e.evaluation_case_id
                WHERE l.execution_case_id = ?
                ORDER BY e.overall_score DESC, e.created_at DESC
                LIMIT ?
                """,
                (execution_case_id, int(limit)),
            ).fetchall()
        return [SkillEvaluationCase.model_validate(json.loads(r["payload_json"])) for r in rows]

