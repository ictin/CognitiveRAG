from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable


def _json(payload: str | None) -> dict:
    if not payload:
        return {}
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _skill_db_candidates(workdir: str | Path, filename: str) -> list[Path]:
    root = Path(workdir)
    return [
        root / "skill_memory" / filename,
        root / "data" / "skill_memory" / filename,
        root / filename,
    ]


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _ensure_reasoning_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(reasoning_patterns)").fetchall()}
    if "success_signal_count" not in cols:
        conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN success_signal_count INTEGER NOT NULL DEFAULT 0")
    if "failure_signal_count" not in cols:
        conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN failure_signal_count INTEGER NOT NULL DEFAULT 0")
    if "success_confidence" not in cols:
        conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN success_confidence REAL NOT NULL DEFAULT 0.0")
    if "success_basis_json" not in cols:
        conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN success_basis_json TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_success_conf ON reasoning_patterns(success_confidence)")


def _resolve_success_confidence(*, base_confidence: float, success_count: int, failure_count: int) -> float:
    # Bounded additive signal from repeated success; failures reduce gain.
    success_bonus = min(0.30, float(max(0, success_count)) * 0.07)
    failure_penalty = min(0.20, float(max(0, failure_count)) * 0.06)
    value = max(0.0, min(1.0, float(base_confidence) + success_bonus - failure_penalty))
    return round(value, 6)


def refresh_reasoning_success_signals(*, workdir: str | Path, reasoning_db_path: str | Path | None = None) -> dict:
    """Recompute per-pattern repeated-success signals from stored execution/evaluation cases.

    Deterministic and bounded:
    - success source: unique execution_case_id where execution success OR linked evaluation pass
    - failure source: unique execution_case_id with explicit fail and no pass for that execution
    - missing/ambiguous data does not count as success
    """
    root = Path(workdir)
    reasoning_db = Path(reasoning_db_path) if reasoning_db_path else (root / "reasoning.sqlite3")
    if not reasoning_db.exists():
        return {"updated_patterns": 0, "reason": "reasoning_db_missing"}

    exec_db = _first_existing(_skill_db_candidates(root, "skill_execution.sqlite3"))
    eval_db = _first_existing(_skill_db_candidates(root, "skill_evaluation.sqlite3"))
    if exec_db is None and eval_db is None:
        return {"updated_patterns": 0, "reason": "skill_signal_db_missing"}

    with sqlite3.connect(reasoning_db) as rconn:
        rconn.row_factory = sqlite3.Row
        _ensure_reasoning_columns(rconn)
        pattern_rows = rconn.execute(
            "SELECT pattern_id, canonical_pattern_id, confidence FROM reasoning_patterns"
        ).fetchall()
        if not pattern_rows:
            return {"updated_patterns": 0, "reason": "no_reasoning_patterns"}

        canonical_by_pattern: dict[str, str] = {}
        base_conf_by_canonical: dict[str, float] = {}
        for row in pattern_rows:
            pid = str(row["pattern_id"])
            canonical = str(row["canonical_pattern_id"] or pid)
            canonical_by_pattern[pid] = canonical
            prev = base_conf_by_canonical.get(canonical, 0.0)
            base_conf_by_canonical[canonical] = max(prev, float(row["confidence"] or 0.0))

        success_exec_ids: dict[str, set[str]] = {c: set() for c in base_conf_by_canonical}
        fail_exec_ids: dict[str, set[str]] = {c: set() for c in base_conf_by_canonical}

        execution_artifacts: dict[str, list[str]] = {}
        if exec_db and exec_db.exists():
            with sqlite3.connect(exec_db) as econn:
                econn.row_factory = sqlite3.Row
                rows = econn.execute("SELECT payload_json FROM skill_execution_cases").fetchall()
                for row in rows:
                    payload = _json(row["payload_json"])
                    exec_id = str(payload.get("execution_case_id") or "").strip()
                    if not exec_id:
                        continue
                    artifact_ids = [str(a).strip() for a in payload.get("selected_artifact_ids", []) if str(a).strip()]
                    execution_artifacts[exec_id] = artifact_ids
                    if bool(payload.get("success_flag", False)):
                        for artifact_id in artifact_ids:
                            canonical = canonical_by_pattern.get(artifact_id)
                            if canonical:
                                success_exec_ids[canonical].add(exec_id)
                    else:
                        for artifact_id in artifact_ids:
                            canonical = canonical_by_pattern.get(artifact_id)
                            if canonical:
                                fail_exec_ids[canonical].add(exec_id)

        if eval_db and eval_db.exists():
            with sqlite3.connect(eval_db) as vconn:
                vconn.row_factory = sqlite3.Row
                rows = vconn.execute("SELECT payload_json FROM skill_evaluation_cases").fetchall()
                for row in rows:
                    payload = _json(row["payload_json"])
                    exec_id = str(payload.get("execution_case_id") or "").strip()
                    if not exec_id:
                        continue
                    artifact_ids = execution_artifacts.get(exec_id, [])
                    if not artifact_ids:
                        continue
                    pass_flag = bool(payload.get("pass_flag", False))
                    overall = float(payload.get("overall_score", 0.0) or 0.0)
                    is_success = pass_flag and overall >= 0.7
                    is_fail = (not pass_flag) or overall < 0.5
                    for artifact_id in artifact_ids:
                        canonical = canonical_by_pattern.get(artifact_id)
                        if not canonical:
                            continue
                        if is_success:
                            success_exec_ids[canonical].add(exec_id)
                        elif is_fail:
                            fail_exec_ids[canonical].add(exec_id)

        updated = 0
        for canonical, base_conf in base_conf_by_canonical.items():
            success_ids = sorted(success_exec_ids.get(canonical, set()))
            fail_ids = sorted(fail_exec_ids.get(canonical, set()) - set(success_ids))
            success_count = len(success_ids)
            fail_count = len(fail_ids)
            success_conf = _resolve_success_confidence(
                base_confidence=base_conf,
                success_count=success_count,
                failure_count=fail_count,
            )
            basis = {
                "source": "execution_evaluation_memory",
                "success_execution_case_ids": success_ids[:20],
                "failure_execution_case_ids": fail_ids[:20],
                "success_signal_count": success_count,
                "failure_signal_count": fail_count,
                "base_confidence": round(base_conf, 6),
                "resolved_success_confidence": success_conf,
            }
            rconn.execute(
                """
                UPDATE reasoning_patterns
                SET success_signal_count=?,
                    failure_signal_count=?,
                    success_confidence=?,
                    success_basis_json=?
                WHERE canonical_pattern_id=? OR pattern_id=?
                """,
                (
                    success_count,
                    fail_count,
                    success_conf,
                    json.dumps(basis, sort_keys=True),
                    canonical,
                    canonical,
                ),
            )
            updated += int(rconn.total_changes > 0)

    return {
        "updated_patterns": len(base_conf_by_canonical),
        "execution_db_used": str(exec_db) if exec_db else "",
        "evaluation_db_used": str(eval_db) if eval_db else "",
    }

