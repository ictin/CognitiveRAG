from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

from CognitiveRAG.crag.graph_memory.skill_graph import record_execution_case_graph_links
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.skill_memory.case_linker import normalize_artifact_ids
from CognitiveRAG.crag.skill_memory.execution_schema import SkillExecutionCase


class SkillExecutionStore:
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
                CREATE TABLE IF NOT EXISTS skill_execution_cases (
                    execution_case_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    channel_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    selected_artifact_ids_json TEXT NOT NULL,
                    pack_summary TEXT NOT NULL,
                    pack_ref TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    output_ref TEXT NOT NULL,
                    success_flag INTEGER NOT NULL,
                    human_edits_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_execution_case_artifacts (
                    execution_case_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    PRIMARY KEY (execution_case_id, artifact_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_agent_task ON skill_execution_cases(agent_type, task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_channel ON skill_execution_cases(channel_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_created ON skill_execution_cases(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_link_artifact ON skill_execution_case_artifacts(artifact_id)")

    def upsert_case(self, case: SkillExecutionCase) -> None:
        payload = case.model_dump()
        artifact_ids = normalize_artifact_ids(case.selected_artifact_ids)
        payload["selected_artifact_ids"] = artifact_ids
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skill_execution_cases(
                    execution_case_id, agent_type, task_type, channel_type, language, request_text,
                    selected_artifact_ids_json, pack_summary, pack_ref, output_text, output_ref,
                    success_flag, human_edits_json, notes, created_at, updated_at, provenance_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_case_id) DO UPDATE SET
                    agent_type=excluded.agent_type,
                    task_type=excluded.task_type,
                    channel_type=excluded.channel_type,
                    language=excluded.language,
                    request_text=excluded.request_text,
                    selected_artifact_ids_json=excluded.selected_artifact_ids_json,
                    pack_summary=excluded.pack_summary,
                    pack_ref=excluded.pack_ref,
                    output_text=excluded.output_text,
                    output_ref=excluded.output_ref,
                    success_flag=excluded.success_flag,
                    human_edits_json=excluded.human_edits_json,
                    notes=excluded.notes,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    provenance_json=excluded.provenance_json,
                    payload_json=excluded.payload_json
                """,
                (
                    case.execution_case_id,
                    case.agent_type,
                    case.task_type,
                    case.channel_type,
                    case.language,
                    case.request_text,
                    json.dumps(artifact_ids),
                    case.pack_summary,
                    case.pack_ref,
                    case.output_text,
                    case.output_ref,
                    int(case.success_flag),
                    json.dumps(case.human_edits),
                    case.notes,
                    case.created_at,
                    case.updated_at,
                    json.dumps(case.provenance.model_dump()),
                    json.dumps(payload),
                ),
            )
            conn.execute(
                "DELETE FROM skill_execution_case_artifacts WHERE execution_case_id = ?",
                (case.execution_case_id,),
            )
            conn.executemany(
                """
                INSERT OR IGNORE INTO skill_execution_case_artifacts(execution_case_id, artifact_id)
                VALUES (?, ?)
                """,
                [(case.execution_case_id, artifact_id) for artifact_id in artifact_ids],
            )
        graph_store = GraphMemoryStore(self.db_path.parent / "graph_memory.sqlite3")
        record_execution_case_graph_links(graph_store, case=case)

    def upsert_many(self, cases: Iterable[SkillExecutionCase]) -> int:
        count = 0
        for case in cases:
            self.upsert_case(case)
            count += 1
        return count

    def get_case(self, execution_case_id: str) -> SkillExecutionCase | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM skill_execution_cases WHERE execution_case_id = ?",
                (execution_case_id,),
            ).fetchone()
        if not row:
            return None
        return SkillExecutionCase.model_validate(json.loads(row["payload_json"]))

    def list_cases(
        self,
        *,
        agent_type: str | None = None,
        task_type: str | None = None,
        channel_type: str | None = None,
        limit: int = 20,
    ) -> List[SkillExecutionCase]:
        where_clauses: List[str] = []
        params: List[object] = []
        if agent_type:
            where_clauses.append("agent_type = ?")
            params.append(agent_type)
        if task_type:
            where_clauses.append("task_type = ?")
            params.append(task_type)
        if channel_type:
            where_clauses.append("channel_type = ?")
            params.append(channel_type)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload_json FROM skill_execution_cases
                WHERE {where_sql}
                ORDER BY created_at DESC, execution_case_id
                LIMIT ?
                """,
                (*params, int(limit)),
            ).fetchall()
        return [SkillExecutionCase.model_validate(json.loads(r["payload_json"])) for r in rows]
