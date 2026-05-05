from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_id(prefix: str, payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


class RuntimeTuningStore:
    STATUS_DRAFT = "draft"
    STATUS_APPROVED = "approved"
    STATUS_REJECTED = "rejected"

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
                CREATE TABLE IF NOT EXISTS research_evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    scenario_ids_json TEXT NOT NULL,
                    artifact_refs_json TEXT NOT NULL,
                    metrics_before_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_tuning_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    authoritative INTEGER NOT NULL,
                    requires_explicit_acceptance INTEGER NOT NULL,
                    candidate_patch_json TEXT NOT NULL,
                    metrics_after_json TEXT NOT NULL,
                    comparison_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    rollback_state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def create_evaluation_run(
        self,
        *,
        scenario_ids: List[str],
        artifact_refs: List[str],
        metrics_before: Dict[str, float],
        summary: str,
    ) -> Dict[str, Any]:
        payload_seed = {
            "scenario_ids": sorted([str(x) for x in scenario_ids]),
            "artifact_refs": sorted([str(x) for x in artifact_refs]),
            "metrics_before": metrics_before,
            "summary": str(summary or "").strip(),
        }
        run_id = _stable_id("evalrun", payload_seed)
        payload = {
            "run_id": run_id,
            "scenario_ids": payload_seed["scenario_ids"],
            "artifact_refs": payload_seed["artifact_refs"],
            "metrics_before": dict(metrics_before),
            "summary": payload_seed["summary"],
            "created_at": _now_iso(),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_evaluation_runs(
                    run_id, scenario_ids_json, artifact_refs_json, metrics_before_json, summary, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["run_id"],
                    json.dumps(payload["scenario_ids"]),
                    json.dumps(payload["artifact_refs"]),
                    json.dumps(payload["metrics_before"]),
                    payload["summary"],
                    payload["created_at"],
                    json.dumps(payload),
                ),
            )
        return payload

    def create_candidate_proposal(
        self,
        *,
        run_id: str,
        candidate_patch: Dict[str, Any],
        metrics_after: Dict[str, float],
        rationale: str,
    ) -> Dict[str, Any]:
        run = self.get_evaluation_run(run_id)
        if not run:
            raise ValueError(f"unknown evaluation run_id: {run_id}")

        before = dict(run.get("metrics_before") or {})
        delta = {k: float(metrics_after.get(k, 0.0)) - float(before.get(k, 0.0)) for k in sorted(set(before) | set(metrics_after))}
        comparison = {
            "before": before,
            "after": dict(metrics_after),
            "delta": delta,
            "summary": "candidate proposal is non-authoritative and requires explicit acceptance",
        }
        seed = {
            "run_id": run_id,
            "candidate_patch": candidate_patch,
            "metrics_after": metrics_after,
            "rationale": rationale,
        }
        proposal_id = _stable_id("tuneprop", seed)
        now = _now_iso()
        payload = {
            "proposal_id": proposal_id,
            "run_id": run_id,
            "status": self.STATUS_DRAFT,
            "authoritative": False,
            "requires_explicit_acceptance": True,
            "candidate_patch": dict(candidate_patch),
            "metrics_after": dict(metrics_after),
            "comparison": comparison,
            "rationale": str(rationale or "").strip(),
            "rollback_state": {
                "runtime_changed": False,
                "rollback_ready": True,
                "reason": "proposal_not_applied",
            },
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_tuning_proposals(
                    proposal_id, run_id, status, authoritative, requires_explicit_acceptance,
                    candidate_patch_json, metrics_after_json, comparison_json, rationale,
                    rollback_state_json, created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    run_id,
                    payload["status"],
                    0,
                    1,
                    json.dumps(payload["candidate_patch"]),
                    json.dumps(payload["metrics_after"]),
                    json.dumps(payload["comparison"]),
                    payload["rationale"],
                    json.dumps(payload["rollback_state"]),
                    payload["created_at"],
                    payload["updated_at"],
                    json.dumps(payload),
                ),
            )
        return payload

    def reject_proposal(self, proposal_id: str, *, reason: str) -> Dict[str, Any]:
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"unknown proposal_id: {proposal_id}")
        proposal["status"] = self.STATUS_REJECTED
        proposal["authoritative"] = False
        proposal["updated_at"] = _now_iso()
        proposal["rollback_state"] = {
            "runtime_changed": False,
            "rollback_ready": True,
            "reason": str(reason or "rejected").strip(),
        }
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE research_tuning_proposals
                SET status=?, authoritative=0, rollback_state_json=?, updated_at=?, payload_json=?
                WHERE proposal_id=?
                """,
                (
                    proposal["status"],
                    json.dumps(proposal["rollback_state"]),
                    proposal["updated_at"],
                    json.dumps(proposal),
                    proposal_id,
                ),
            )
        return proposal

    def get_evaluation_run(self, run_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM research_evaluation_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["payload_json"])

    def get_proposal(self, proposal_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM research_tuning_proposals WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["payload_json"])

    def get_effective_runtime_policy(self) -> Dict[str, Any]:
        # Runtime remains unchanged by draft/rejected proposals in this slice.
        return {
            "source": "baseline",
            "applied_proposal_id": None,
            "policy": {
                "selector_authority": "default",
                "retrieval_ranking": "default",
                "discovery_bounds": "default",
            },
        }
