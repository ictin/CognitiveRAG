from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class WebPromotedMemoryStore:
    STATE_STAGED = "staged"
    STATE_TRUSTED = "trusted"
    FRESHNESS_FRESH = "fresh"
    FRESHNESS_STALE = "stale"
    FRESHNESS_REVALIDATION_PENDING = "revalidation_pending"
    DEFAULT_TTL_HOURS = 72.0

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
                    freshness_lifecycle_state TEXT NOT NULL DEFAULT 'fresh',
                    freshness_reason TEXT NOT NULL DEFAULT '',
                    freshness_policy_json TEXT NOT NULL DEFAULT '{}',
                    last_validated_at TEXT,
                    revalidation_requested_at TEXT,
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
            if "freshness_lifecycle_state" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN freshness_lifecycle_state TEXT NOT NULL DEFAULT 'fresh'")
            if "freshness_reason" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN freshness_reason TEXT NOT NULL DEFAULT ''")
            if "freshness_policy_json" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN freshness_policy_json TEXT NOT NULL DEFAULT '{}'")
            if "last_validated_at" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN last_validated_at TEXT")
            if "revalidation_requested_at" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN revalidation_requested_at TEXT")
            if "approval_reason" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN approval_reason TEXT NOT NULL DEFAULT ''")
            if "approval_basis_json" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN approval_basis_json TEXT NOT NULL DEFAULT '{}'")
            if "approved_at" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN approved_at TEXT")
            if "state_updated_at" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN state_updated_at TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_promoted_state ON web_promoted_memory(promotion_state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_promoted_freshness_lifecycle ON web_promoted_memory(freshness_lifecycle_state)")
            # Conservative migration guardrail: rows without any validation basis should not stay fresh forever.
            conn.execute(
                """
                UPDATE web_promoted_memory
                SET freshness_lifecycle_state='stale',
                    freshness_reason=CASE
                        WHEN freshness_reason IS NULL OR freshness_reason='' THEN 'missing_validation_basis'
                        ELSE freshness_reason
                    END
                WHERE (last_validated_at IS NULL OR last_validated_at='')
                  AND (approved_at IS NULL OR approved_at='')
                  AND (state_updated_at IS NULL OR state_updated_at='')
                """
            )

    @staticmethod
    def _state(state: str | None) -> str:
        s = str(state or "").strip().lower()
        return s if s in {WebPromotedMemoryStore.STATE_STAGED, WebPromotedMemoryStore.STATE_TRUSTED} else WebPromotedMemoryStore.STATE_STAGED

    @staticmethod
    def _parse_iso(ts: str | None) -> datetime | None:
        raw = str(ts or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _freshness_state(state: str | None) -> str:
        s = str(state or "").strip().lower()
        valid = {
            WebPromotedMemoryStore.FRESHNESS_FRESH,
            WebPromotedMemoryStore.FRESHNESS_STALE,
            WebPromotedMemoryStore.FRESHNESS_REVALIDATION_PENDING,
        }
        return s if s in valid else WebPromotedMemoryStore.FRESHNESS_STALE

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
        freshness_lifecycle_state: str,
        freshness_reason: str,
        freshness_policy: Dict[str, Any] | None,
        last_validated_at: str | None,
        revalidation_requested_at: str | None,
    ) -> None:
        state = self._state(promotion_state)
        lifecycle = self._freshness_state(freshness_lifecycle_state)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_promoted_memory(
                    promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state,
                    promotion_state, freshness_lifecycle_state, freshness_reason, freshness_policy_json,
                    last_validated_at, revalidation_requested_at,
                    approval_reason, approval_basis_json, approved_at, state_updated_at,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(promoted_id) DO UPDATE SET
                    canonical_fact=excluded.canonical_fact,
                    evidence_ids_json=excluded.evidence_ids_json,
                    confidence=excluded.confidence,
                    freshness_state=excluded.freshness_state,
                    promotion_state=excluded.promotion_state,
                    freshness_lifecycle_state=excluded.freshness_lifecycle_state,
                    freshness_reason=excluded.freshness_reason,
                    freshness_policy_json=excluded.freshness_policy_json,
                    last_validated_at=excluded.last_validated_at,
                    revalidation_requested_at=excluded.revalidation_requested_at,
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
                    lifecycle,
                    freshness_reason or "",
                    json.dumps(freshness_policy or {}),
                    last_validated_at,
                    revalidation_requested_at,
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
        freshness_lifecycle_state: str = FRESHNESS_FRESH,
        freshness_reason: str = "",
        freshness_policy: Dict[str, Any] | None = None,
        last_validated_at: str | None = None,
        revalidation_requested_at: str | None = None,
    ) -> None:
        now = now_iso or ""
        state = self._state(promotion_state)
        lifecycle = self._freshness_state(freshness_lifecycle_state)
        effective_last_validated = last_validated_at if last_validated_at is not None else (now if lifecycle == self.FRESHNESS_FRESH else None)
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
            freshness_lifecycle_state=lifecycle,
            freshness_reason=freshness_reason,
            freshness_policy=(freshness_policy or {}),
            last_validated_at=effective_last_validated,
            revalidation_requested_at=revalidation_requested_at,
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
            freshness_lifecycle_state=self.FRESHNESS_STALE,
            freshness_reason="staged_not_yet_validated",
            freshness_policy={"ttl_hours": self.DEFAULT_TTL_HOURS},
            last_validated_at=None,
            revalidation_requested_at=now_iso,
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
            freshness_lifecycle_state=(self.FRESHNESS_FRESH if eligible else self.FRESHNESS_STALE),
            freshness_reason=("approved_and_validated" if eligible else "insufficient_evidence_or_confidence"),
            freshness_policy={"ttl_hours": self.DEFAULT_TTL_HOURS},
            last_validated_at=(now if eligible else record.get("last_validated_at")),
            revalidation_requested_at=(None if eligible else record.get("revalidation_requested_at")),
        )
        return self.get(promoted_id)

    def evaluate_freshness(self, promoted_id: str, *, now_iso: str | None = None) -> Dict[str, Any] | None:
        record = self.get(promoted_id)
        if not record:
            return None
        if record.get("promotion_state") != self.STATE_TRUSTED:
            return record

        now = self._parse_iso(now_iso) or datetime.now(timezone.utc)
        md = dict(record.get("metadata") or {})
        policy = dict(record.get("freshness_policy") or {})
        ttl_hours = float(md.get("freshness_ttl_hours") or policy.get("ttl_hours") or self.DEFAULT_TTL_HOURS)
        last_validated = self._parse_iso(record.get("last_validated_at"))
        if not last_validated:
            last_validated = self._parse_iso(record.get("approved_at"))
        if not last_validated:
            last_validated = self._parse_iso(record.get("updated_at"))
        if not last_validated:
            last_validated = self._parse_iso(record.get("created_at"))

        if not last_validated:
            lifecycle = self.FRESHNESS_STALE
            reason = "missing_validation_basis"
        else:
            age_hours = (now - last_validated).total_seconds() / 3600.0
            lifecycle = self.FRESHNESS_FRESH if age_hours <= ttl_hours else self.FRESHNESS_STALE
            reason = (
                "within_ttl_window"
                if lifecycle == self.FRESHNESS_FRESH
                else "ttl_expired"
            )

        # Keep read-time freshness checks deterministic; avoid mutating rows unless
        # the lifecycle decision has actually changed.
        if (
            str(record.get("freshness_lifecycle_state") or "") == lifecycle
            and str(record.get("freshness_reason") or "") == reason
            and float((record.get("freshness_policy") or {}).get("ttl_hours") or self.DEFAULT_TTL_HOURS) == ttl_hours
        ):
            return record

        self._upsert_internal(
            promoted_id=record["promoted_id"],
            canonical_fact=record["canonical_fact"],
            evidence_ids=list(record.get("evidence_ids") or []),
            confidence=float(record.get("confidence") or 0.0),
            freshness_state=record["freshness_state"],
            metadata=md,
            now_iso=(now_iso or now.isoformat().replace("+00:00", "Z")),
            promotion_state=record["promotion_state"],
            approval_reason=record.get("approval_reason") or "",
            approval_basis=dict(record.get("approval_basis") or {}),
            approved_at=record.get("approved_at"),
            freshness_lifecycle_state=lifecycle,
            freshness_reason=reason,
            freshness_policy={"ttl_hours": ttl_hours},
            last_validated_at=(record.get("last_validated_at") or record.get("approved_at")),
            revalidation_requested_at=record.get("revalidation_requested_at"),
        )
        return self.get(promoted_id)

    def request_revalidation(self, promoted_id: str, *, reason: str = "stale_requires_revalidation", now_iso: str | None = None) -> Dict[str, Any] | None:
        record = self.get(promoted_id)
        if not record:
            return None
        if record.get("revalidation_requested_at"):
            # Preserve first request timestamp to keep repeated retrieval deterministic.
            return record
        now = now_iso or ""
        self._upsert_internal(
            promoted_id=record["promoted_id"],
            canonical_fact=record["canonical_fact"],
            evidence_ids=list(record.get("evidence_ids") or []),
            confidence=float(record.get("confidence") or 0.0),
            freshness_state=record["freshness_state"],
            metadata=dict(record.get("metadata") or {}),
            now_iso=now,
            promotion_state=record["promotion_state"],
            approval_reason=record.get("approval_reason") or "",
            approval_basis=dict(record.get("approval_basis") or {}),
            approved_at=record.get("approved_at"),
            freshness_lifecycle_state=self.FRESHNESS_REVALIDATION_PENDING,
            freshness_reason=reason,
            freshness_policy=dict(record.get("freshness_policy") or {"ttl_hours": self.DEFAULT_TTL_HOURS}),
            last_validated_at=record.get("last_validated_at"),
            revalidation_requested_at=now,
        )
        return self.get(promoted_id)

    def get(self, promoted_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state, promotion_state,
                       freshness_lifecycle_state, freshness_reason, freshness_policy_json, last_validated_at, revalidation_requested_at,
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
        try:
            freshness_policy = json.loads(row["freshness_policy_json"]) if row["freshness_policy_json"] else {}
        except Exception:
            freshness_policy = {}
        return {
            "promoted_id": row["promoted_id"],
            "canonical_fact": row["canonical_fact"],
            "evidence_ids": evidence_ids,
            "confidence": row["confidence"],
            "freshness_state": row["freshness_state"],
            "promotion_state": self._state(row["promotion_state"]),
            "freshness_lifecycle_state": self._freshness_state(row["freshness_lifecycle_state"]),
            "freshness_reason": row["freshness_reason"] or "",
            "freshness_policy": freshness_policy,
            "last_validated_at": row["last_validated_at"],
            "revalidation_requested_at": row["revalidation_requested_at"],
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
                           freshness_lifecycle_state, freshness_reason, freshness_policy_json, last_validated_at, revalidation_requested_at,
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
                           freshness_lifecycle_state, freshness_reason, freshness_policy_json, last_validated_at, revalidation_requested_at,
                           approval_reason, approval_basis_json, approved_at, state_updated_at, metadata_json, created_at, updated_at
                    FROM web_promoted_memory
                    WHERE canonical_fact LIKE ?
                    ORDER BY CASE WHEN promotion_state='trusted' THEN 0 ELSE 1 END,
                             CASE WHEN freshness_lifecycle_state='fresh' THEN 0 WHEN freshness_lifecycle_state='revalidation_pending' THEN 1 ELSE 2 END,
                             confidence DESC, updated_at DESC
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
            try:
                freshness_policy = json.loads(r["freshness_policy_json"]) if r["freshness_policy_json"] else {}
            except Exception:
                freshness_policy = {}
            out.append(
                {
                    "promoted_id": r["promoted_id"],
                    "canonical_fact": r["canonical_fact"],
                    "evidence_ids": evidence_ids,
                    "confidence": r["confidence"],
                    "freshness_state": r["freshness_state"],
                    "promotion_state": self._state(r["promotion_state"]),
                    "freshness_lifecycle_state": self._freshness_state(r["freshness_lifecycle_state"]),
                    "freshness_reason": r["freshness_reason"] or "",
                    "freshness_policy": freshness_policy,
                    "last_validated_at": r["last_validated_at"],
                    "revalidation_requested_at": r["revalidation_requested_at"],
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
