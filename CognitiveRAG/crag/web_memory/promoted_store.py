from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class WebPromotedMemoryStore:
    STATE_STAGED = "staged"
    STATE_TRUSTED = "trusted"

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
                CREATE TABLE IF NOT EXISTS web_promoted_memory (
                    promoted_id TEXT PRIMARY KEY,
                    canonical_fact TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    freshness_state TEXT NOT NULL,
                    promotion_state TEXT NOT NULL DEFAULT 'trusted',
                    approval_reason TEXT NOT NULL DEFAULT '',
                    approval_basis_json TEXT NOT NULL DEFAULT '{}',
                    approved_at TEXT,
                    state_updated_at TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(web_promoted_memory)").fetchall()}
            if "promotion_state" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN promotion_state TEXT NOT NULL DEFAULT 'trusted'")
            if "approval_reason" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN approval_reason TEXT NOT NULL DEFAULT ''")
            if "approval_basis_json" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN approval_basis_json TEXT NOT NULL DEFAULT '{}'")
            if "approved_at" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN approved_at TEXT")
            if "state_updated_at" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN state_updated_at TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_promoted_state ON web_promoted_memory(promotion_state)")

    @staticmethod
    def _state(state: str | None) -> str:
        s = str(state or "").strip().lower()
        return s if s in {WebPromotedMemoryStore.STATE_STAGED, WebPromotedMemoryStore.STATE_TRUSTED} else WebPromotedMemoryStore.STATE_STAGED

    def _evaluate_transition(
        self,
        *,
        evidence_ids: List[str],
        confidence: float,
        metadata: Dict[str, Any] | None,
    ) -> tuple[bool, str, Dict[str, Any]]:
        md = dict(metadata or {})
        unique_ids = sorted({str(e).strip() for e in (evidence_ids or []) if str(e).strip()})
        explicit = bool(md.get("backend_approval"))
        has_source = bool(md.get("source_url") or md.get("source_id"))
        conf = float(confidence or 0.0)
        eligible = explicit or (len(unique_ids) >= 2 and conf >= 0.7 and has_source)
        reason = (
            "explicit_backend_approval"
            if explicit
            else ("evidence_count_confidence_source_threshold" if eligible else "insufficient_evidence_or_confidence")
        )
        basis = {
            "evidence_count": len(unique_ids),
            "confidence": conf,
            "has_source": has_source,
            "explicit_backend_approval": explicit,
        }
        return eligible, reason, basis

    def _upsert_internal(
        self,
        *,
        promoted_id: str,
        canonical_fact: str,
        evidence_ids: List[str],
        confidence: float,
        freshness_state: str,
        metadata: Dict[str, Any] | None,
        now_iso: str,
        promotion_state: str,
        approval_reason: str,
        approval_basis: Dict[str, Any] | None,
        approved_at: str | None,
    ) -> None:
        state = self._state(promotion_state)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_promoted_memory(
                    promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state,
                    promotion_state, approval_reason, approval_basis_json, approved_at, state_updated_at,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(promoted_id) DO UPDATE SET
                    canonical_fact=excluded.canonical_fact,
                    evidence_ids_json=excluded.evidence_ids_json,
                    confidence=excluded.confidence,
                    freshness_state=excluded.freshness_state,
                    promotion_state=excluded.promotion_state,
                    approval_reason=excluded.approval_reason,
                    approval_basis_json=excluded.approval_basis_json,
                    approved_at=excluded.approved_at,
                    state_updated_at=excluded.state_updated_at,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    promoted_id,
                    canonical_fact,
                    json.dumps(list(evidence_ids or [])),
                    float(confidence),
                    freshness_state,
                    state,
                    approval_reason or "",
                    json.dumps(approval_basis or {}),
                    approved_at,
                    now_iso,
                    json.dumps(metadata or {}),
                    now_iso,
                    now_iso,
                ),
            )

    def upsert_fact(
        self,
        *,
        promoted_id: str,
        canonical_fact: str,
        evidence_ids: List[str],
        confidence: float,
        freshness_state: str,
        metadata: Dict[str, Any] | None = None,
        now_iso: str | None = None,
        promotion_state: str = STATE_TRUSTED,
        approval_reason: str = "",
        approval_basis: Dict[str, Any] | None = None,
    ) -> None:
        now = now_iso or ""
        state = self._state(promotion_state)
        self._upsert_internal(
            promoted_id=promoted_id,
            canonical_fact=canonical_fact,
            evidence_ids=evidence_ids,
            confidence=confidence,
            freshness_state=freshness_state,
            metadata=metadata,
            now_iso=now,
            promotion_state=state,
            approval_reason=approval_reason if state == self.STATE_TRUSTED else "staged_pending_approval",
            approval_basis=(approval_basis or {}),
            approved_at=(now if state == self.STATE_TRUSTED else None),
        )

    def stage_fact(
        self,
        *,
        promoted_id: str,
        canonical_fact: str,
        evidence_ids: List[str],
        confidence: float,
        freshness_state: str,
        metadata: Dict[str, Any] | None = None,
        now_iso: str | None = None,
    ) -> None:
        self.upsert_fact(
            promoted_id=promoted_id,
            canonical_fact=canonical_fact,
            evidence_ids=evidence_ids,
            confidence=confidence,
            freshness_state=freshness_state,
            metadata=metadata,
            now_iso=now_iso,
            promotion_state=self.STATE_STAGED,
            approval_reason="staged_pending_approval",
            approval_basis={"stage_source": "web_evidence"},
        )

    def promote_if_eligible(self, promoted_id: str, *, now_iso: str | None = None) -> Dict[str, Any] | None:
        record = self.get(promoted_id)
        if not record:
            return None
        now = now_iso or ""
        evidence_ids = list(record.get("evidence_ids") or [])
        confidence = float(record.get("confidence") or 0.0)
        metadata = dict(record.get("metadata") or {})
        eligible, reason, basis = self._evaluate_transition(
            evidence_ids=evidence_ids,
            confidence=confidence,
            metadata=metadata,
        )
        next_state = self.STATE_TRUSTED if eligible else self.STATE_STAGED
        self._upsert_internal(
            promoted_id=record["promoted_id"],
            canonical_fact=record["canonical_fact"],
            evidence_ids=evidence_ids,
            confidence=confidence,
            freshness_state=record["freshness_state"],
            metadata=metadata,
            now_iso=now,
            promotion_state=next_state,
            approval_reason=reason,
            approval_basis=basis,
            approved_at=(now if eligible else None),
        )
        return self.get(promoted_id)

    def get(self, promoted_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state, promotion_state,
                       approval_reason, approval_basis_json, approved_at, state_updated_at, metadata_json, created_at, updated_at
                FROM web_promoted_memory
                WHERE promoted_id=?
                """,
                (promoted_id,),
            ).fetchone()
        if not row:
            return None
        try:
            evidence_ids = json.loads(row["evidence_ids_json"]) if row["evidence_ids_json"] else []
        except Exception:
            evidence_ids = []
        try:
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        except Exception:
            metadata = {}
        try:
            approval_basis = json.loads(row["approval_basis_json"]) if row["approval_basis_json"] else {}
        except Exception:
            approval_basis = {}
        return {
            "promoted_id": row["promoted_id"],
            "canonical_fact": row["canonical_fact"],
            "evidence_ids": evidence_ids,
            "confidence": row["confidence"],
            "freshness_state": row["freshness_state"],
            "promotion_state": self._state(row["promotion_state"]),
            "approval_reason": row["approval_reason"] or "",
            "approval_basis": approval_basis,
            "approved_at": row["approved_at"],
            "state_updated_at": row["state_updated_at"],
            "metadata": metadata,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def search(self, query: str, top_k: int = 5, state: str | None = None) -> List[Dict[str, Any]]:
        q = f"%{(query or '').strip()}%"
        state_filter = self._state(state) if state else None
        with self._connect() as conn:
            if state_filter:
                rows = conn.execute(
                    """
                    SELECT promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state, promotion_state,
                           approval_reason, approval_basis_json, approved_at, state_updated_at, metadata_json, created_at, updated_at
                    FROM web_promoted_memory
                    WHERE canonical_fact LIKE ? AND promotion_state = ?
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (q, state_filter, int(top_k)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state, promotion_state,
                           approval_reason, approval_basis_json, approved_at, state_updated_at, metadata_json, created_at, updated_at
                    FROM web_promoted_memory
                    WHERE canonical_fact LIKE ?
                    ORDER BY CASE WHEN promotion_state='trusted' THEN 0 ELSE 1 END, confidence DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (q, int(top_k)),
                ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                evidence_ids = json.loads(r["evidence_ids_json"]) if r["evidence_ids_json"] else []
            except Exception:
                evidence_ids = []
            try:
                metadata = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                metadata = {}
            try:
                approval_basis = json.loads(r["approval_basis_json"]) if r["approval_basis_json"] else {}
            except Exception:
                approval_basis = {}
            out.append(
                {
                    "promoted_id": r["promoted_id"],
                    "canonical_fact": r["canonical_fact"],
                    "evidence_ids": evidence_ids,
                    "confidence": r["confidence"],
                    "freshness_state": r["freshness_state"],
                    "promotion_state": self._state(r["promotion_state"]),
                    "approval_reason": r["approval_reason"] or "",
                    "approval_basis": approval_basis,
                    "approved_at": r["approved_at"],
                    "state_updated_at": r["state_updated_at"],
                    "metadata": metadata,
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
            )
        return out
