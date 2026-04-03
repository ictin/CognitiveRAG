from __future__ import annotations

import hashlib
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
    CONTRADICTION_RELATION = "contradicts"
    CONTRADICTION_STATUS_OPEN = "open"
    TIER_LOCAL = "local"
    TIER_WORKSPACE = "workspace"
    TIER_GLOBAL = "global"

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
                    promotion_tier TEXT NOT NULL DEFAULT 'workspace',
                    origin_tier TEXT NOT NULL DEFAULT 'workspace',
                    promoted_from_ids_json TEXT NOT NULL DEFAULT '[]',
                    promotion_basis_json TEXT NOT NULL DEFAULT '{}',
                    promotion_history_json TEXT NOT NULL DEFAULT '[]',
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_promoted_contradictions (
                    contradiction_id TEXT PRIMARY KEY,
                    claim_a_id TEXT NOT NULL,
                    claim_b_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL DEFAULT 'contradicts',
                    conflict_status TEXT NOT NULL DEFAULT 'open',
                    detection_rule TEXT NOT NULL,
                    source_basis_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(web_promoted_memory)").fetchall()}
            if "promotion_state" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN promotion_state TEXT NOT NULL DEFAULT 'trusted'")
            if "promotion_tier" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN promotion_tier TEXT NOT NULL DEFAULT 'workspace'")
            if "origin_tier" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN origin_tier TEXT NOT NULL DEFAULT 'workspace'")
            if "promoted_from_ids_json" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN promoted_from_ids_json TEXT NOT NULL DEFAULT '[]'")
            if "promotion_basis_json" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN promotion_basis_json TEXT NOT NULL DEFAULT '{}'")
            if "promotion_history_json" not in cols:
                conn.execute("ALTER TABLE web_promoted_memory ADD COLUMN promotion_history_json TEXT NOT NULL DEFAULT '[]'")
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_promoted_tier ON web_promoted_memory(promotion_tier)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_promoted_freshness_lifecycle ON web_promoted_memory(freshness_lifecycle_state)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_web_promoted_contradiction_pair ON web_promoted_contradictions(claim_a_id, claim_b_id, relation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_promoted_contradiction_claim_a ON web_promoted_contradictions(claim_a_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_promoted_contradiction_claim_b ON web_promoted_contradictions(claim_b_id)")
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
    def _tier(tier: str | None) -> str:
        t = str(tier or "").strip().lower()
        valid = {
            WebPromotedMemoryStore.TIER_LOCAL,
            WebPromotedMemoryStore.TIER_WORKSPACE,
            WebPromotedMemoryStore.TIER_GLOBAL,
        }
        return t if t in valid else WebPromotedMemoryStore.TIER_LOCAL

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

    @staticmethod
    def _normalized_text(value: str | None) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _claim_signature(self, record: Dict[str, Any]) -> tuple[str, str]:
        metadata = dict(record.get("metadata") or {})
        key = self._normalized_text(metadata.get("claim_key"))
        value = self._normalized_text(metadata.get("claim_value"))
        if not key or not value:
            return "", ""
        return key, value

    @staticmethod
    def _pair(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    @classmethod
    def _contradiction_id(cls, claim_a_id: str, claim_b_id: str, relation_type: str) -> str:
        a, b = cls._pair(claim_a_id, claim_b_id)
        raw = f"{a}|{b}|{relation_type}"
        return "wpc:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

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
        promotion_tier: str,
        origin_tier: str,
        promoted_from_ids: List[str],
        promotion_basis: Dict[str, Any] | None,
        promotion_history: List[Dict[str, Any]] | None,
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
        tier = self._tier(promotion_tier)
        origin = self._tier(origin_tier)
        lifecycle = self._freshness_state(freshness_lifecycle_state)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_promoted_memory(
                    promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state,
                    promotion_state, promotion_tier, origin_tier, promoted_from_ids_json, promotion_basis_json, promotion_history_json,
                    freshness_lifecycle_state, freshness_reason, freshness_policy_json,
                    last_validated_at, revalidation_requested_at,
                    approval_reason, approval_basis_json, approved_at, state_updated_at,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(promoted_id) DO UPDATE SET
                    canonical_fact=excluded.canonical_fact,
                    evidence_ids_json=excluded.evidence_ids_json,
                    confidence=excluded.confidence,
                    freshness_state=excluded.freshness_state,
                    promotion_state=excluded.promotion_state,
                    promotion_tier=excluded.promotion_tier,
                    origin_tier=excluded.origin_tier,
                    promoted_from_ids_json=excluded.promoted_from_ids_json,
                    promotion_basis_json=excluded.promotion_basis_json,
                    promotion_history_json=excluded.promotion_history_json,
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
                    tier,
                    origin,
                    json.dumps(sorted({str(x).strip() for x in (promoted_from_ids or []) if str(x).strip()})),
                    json.dumps(promotion_basis or {}),
                    json.dumps(promotion_history or []),
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
        promotion_tier: str = TIER_LOCAL,
        origin_tier: str | None = None,
        promoted_from_ids: List[str] | None = None,
        promotion_basis: Dict[str, Any] | None = None,
        promotion_history: List[Dict[str, Any]] | None = None,
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
        tier = self._tier(promotion_tier)
        origin = self._tier(origin_tier or promotion_tier)
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
            promotion_tier=tier,
            origin_tier=origin,
            promoted_from_ids=list(promoted_from_ids or []),
            promotion_basis=dict(promotion_basis or {}),
            promotion_history=list(promotion_history or []),
            approval_reason=approval_reason if state == self.STATE_TRUSTED else "staged_pending_approval",
            approval_basis=(approval_basis or {}),
            approved_at=(now if state == self.STATE_TRUSTED else None),
            freshness_lifecycle_state=lifecycle,
            freshness_reason=freshness_reason,
            freshness_policy=(freshness_policy or {}),
            last_validated_at=effective_last_validated,
            revalidation_requested_at=revalidation_requested_at,
        )
        self._detect_and_record_contradictions(promoted_id=promoted_id, now_iso=now)

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
            promotion_tier=self.TIER_LOCAL,
            origin_tier=self.TIER_LOCAL,
            promoted_from_ids=[],
            promotion_basis={"transition": "seed_local_stage"},
            promotion_history=[],
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
            promotion_tier=record.get("promotion_tier") or self.TIER_LOCAL,
            origin_tier=record.get("origin_tier") or (record.get("promotion_tier") or self.TIER_LOCAL),
            promoted_from_ids=list(record.get("promoted_from_ids") or []),
            promotion_basis=dict(record.get("promotion_basis") or {}),
            promotion_history=list(record.get("promotion_history") or []),
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
            promotion_tier=record.get("promotion_tier") or self.TIER_LOCAL,
            origin_tier=record.get("origin_tier") or (record.get("promotion_tier") or self.TIER_LOCAL),
            promoted_from_ids=list(record.get("promoted_from_ids") or []),
            promotion_basis=dict(record.get("promotion_basis") or {}),
            promotion_history=list(record.get("promotion_history") or []),
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
            promotion_tier=record.get("promotion_tier") or self.TIER_LOCAL,
            origin_tier=record.get("origin_tier") or (record.get("promotion_tier") or self.TIER_LOCAL),
            promoted_from_ids=list(record.get("promoted_from_ids") or []),
            promotion_basis=dict(record.get("promotion_basis") or {}),
            promotion_history=list(record.get("promotion_history") or []),
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

    def register_contradiction(
        self,
        *,
        claim_a_id: str,
        claim_b_id: str,
        detection_rule: str,
        source_basis: Dict[str, Any] | None = None,
        relation_type: str = CONTRADICTION_RELATION,
        conflict_status: str = CONTRADICTION_STATUS_OPEN,
        now_iso: str | None = None,
    ) -> Dict[str, Any]:
        a, b = self._pair(claim_a_id, claim_b_id)
        now = now_iso or ""
        contradiction_id = self._contradiction_id(a, b, relation_type)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_promoted_contradictions(
                    contradiction_id, claim_a_id, claim_b_id, relation_type,
                    conflict_status, detection_rule, source_basis_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(contradiction_id) DO UPDATE SET
                    conflict_status=excluded.conflict_status,
                    detection_rule=excluded.detection_rule,
                    source_basis_json=excluded.source_basis_json,
                    updated_at=excluded.updated_at
                """,
                (
                    contradiction_id,
                    a,
                    b,
                    relation_type,
                    conflict_status,
                    detection_rule,
                    json.dumps(source_basis or {}),
                    now,
                    now,
                ),
            )
        return {
            "contradiction_id": contradiction_id,
            "claim_a_id": a,
            "claim_b_id": b,
            "relation_type": relation_type,
            "conflict_status": conflict_status,
            "detection_rule": detection_rule,
            "source_basis": dict(source_basis or {}),
            "created_at": now,
            "updated_at": now,
        }

    def _list_all_records(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT promoted_id FROM web_promoted_memory ORDER BY promoted_id ASC").fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            rec = self.get(str(row["promoted_id"]))
            if rec:
                out.append(rec)
        return out

    def _detect_and_record_contradictions(self, *, promoted_id: str, now_iso: str | None = None) -> List[Dict[str, Any]]:
        current = self.get(promoted_id)
        if not current:
            return []
        key, value = self._claim_signature(current)
        if not key or not value:
            return []

        now = now_iso or ""
        current_meta = dict(current.get("metadata") or {})
        current_source_class = str(current_meta.get("source_class") or "web_promoted")
        records = self._list_all_records()
        created: List[Dict[str, Any]] = []
        for other in records:
            other_id = str(other.get("promoted_id") or "")
            if not other_id or other_id == promoted_id:
                continue
            other_key, other_value = self._claim_signature(other)
            if not other_key or not other_value:
                continue
            if other_key != key:
                continue
            if other_value == value:
                continue

            other_meta = dict(other.get("metadata") or {})
            basis = {
                "claim_key": key,
                "claim_a_value": value,
                "claim_b_value": other_value,
                "claim_a_source_class": current_source_class,
                "claim_b_source_class": str(other_meta.get("source_class") or "web_promoted"),
                "claim_a_promotion_state": current.get("promotion_state"),
                "claim_b_promotion_state": other.get("promotion_state"),
                "claim_a_freshness_lifecycle_state": current.get("freshness_lifecycle_state"),
                "claim_b_freshness_lifecycle_state": other.get("freshness_lifecycle_state"),
            }
            created.append(
                self.register_contradiction(
                    claim_a_id=promoted_id,
                    claim_b_id=other_id,
                    detection_rule="claim_key_value_mismatch",
                    source_basis=basis,
                    now_iso=now,
                )
            )
        return created

    def get_contradictions_for(self, promoted_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT contradiction_id, claim_a_id, claim_b_id, relation_type, conflict_status, detection_rule, source_basis_json, created_at, updated_at
                FROM web_promoted_contradictions
                WHERE claim_a_id=? OR claim_b_id=?
                ORDER BY updated_at DESC, contradiction_id ASC
                """,
                (promoted_id, promoted_id),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                basis = json.loads(row["source_basis_json"]) if row["source_basis_json"] else {}
            except Exception:
                basis = {}
            claim_a_id = str(row["claim_a_id"] or "")
            claim_b_id = str(row["claim_b_id"] or "")
            other_id = claim_b_id if claim_a_id == promoted_id else claim_a_id
            other = self.get(other_id) if other_id else None
            out.append(
                {
                    "contradiction_id": row["contradiction_id"],
                    "claim_a_id": claim_a_id,
                    "claim_b_id": claim_b_id,
                    "other_claim_id": other_id,
                    "relation_type": row["relation_type"] or self.CONTRADICTION_RELATION,
                    "conflict_status": row["conflict_status"] or self.CONTRADICTION_STATUS_OPEN,
                    "detection_rule": row["detection_rule"] or "",
                    "source_basis": basis,
                    "other_claim_source_class": str((other or {}).get("metadata", {}).get("source_class") or "web_promoted"),
                    "other_claim_promotion_state": (other or {}).get("promotion_state"),
                    "other_claim_freshness_lifecycle_state": (other or {}).get("freshness_lifecycle_state"),
                    "other_claim_last_validated_at": (other or {}).get("last_validated_at"),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return out

    def get_contradiction_summary(self, promoted_id: str) -> Dict[str, Any]:
        contradictions = self.get_contradictions_for(promoted_id)
        open_rows = [row for row in contradictions if str(row.get("conflict_status") or "") == self.CONTRADICTION_STATUS_OPEN]
        other_ids = sorted({str(row.get("other_claim_id") or "") for row in open_rows if str(row.get("other_claim_id") or "")})
        source_classes = sorted(
            {
                str(row.get("other_claim_source_class") or "")
                for row in open_rows
                if str(row.get("other_claim_source_class") or "")
            }
        )
        return {
            "has_contradiction": bool(open_rows),
            "open_contradiction_count": len(open_rows),
            "contradiction_ids": [str(row.get("contradiction_id") or "") for row in open_rows],
            "conflicting_claim_ids": other_ids,
            "conflicting_source_classes": source_classes,
            "contradictions": open_rows,
        }

    def get(self, promoted_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state, promotion_state,
                       promotion_tier, origin_tier, promoted_from_ids_json, promotion_basis_json, promotion_history_json,
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
            promoted_from_ids = json.loads(row["promoted_from_ids_json"]) if row["promoted_from_ids_json"] else []
        except Exception:
            promoted_from_ids = []
        try:
            promotion_basis = json.loads(row["promotion_basis_json"]) if row["promotion_basis_json"] else {}
        except Exception:
            promotion_basis = {}
        try:
            promotion_history = json.loads(row["promotion_history_json"]) if row["promotion_history_json"] else []
        except Exception:
            promotion_history = []
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
            "promotion_tier": self._tier(row["promotion_tier"]),
            "origin_tier": self._tier(row["origin_tier"]),
            "promoted_from_ids": promoted_from_ids,
            "promotion_basis": promotion_basis,
            "promotion_history": promotion_history,
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
                           promotion_tier, origin_tier, promoted_from_ids_json, promotion_basis_json, promotion_history_json,
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
                           promotion_tier, origin_tier, promoted_from_ids_json, promotion_basis_json, promotion_history_json,
                           freshness_lifecycle_state, freshness_reason, freshness_policy_json, last_validated_at, revalidation_requested_at,
                           approval_reason, approval_basis_json, approved_at, state_updated_at, metadata_json, created_at, updated_at
                    FROM web_promoted_memory
                    WHERE canonical_fact LIKE ?
                    ORDER BY CASE WHEN promotion_state='trusted' THEN 0 ELSE 1 END,
                             CASE WHEN promotion_tier='global' THEN 0 WHEN promotion_tier='workspace' THEN 1 ELSE 2 END,
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
                promoted_from_ids = json.loads(r["promoted_from_ids_json"]) if r["promoted_from_ids_json"] else []
            except Exception:
                promoted_from_ids = []
            try:
                promotion_basis = json.loads(r["promotion_basis_json"]) if r["promotion_basis_json"] else {}
            except Exception:
                promotion_basis = {}
            try:
                promotion_history = json.loads(r["promotion_history_json"]) if r["promotion_history_json"] else []
            except Exception:
                promotion_history = []
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
                    "promotion_tier": self._tier(r["promotion_tier"]),
                    "origin_tier": self._tier(r["origin_tier"]),
                    "promoted_from_ids": promoted_from_ids,
                    "promotion_basis": promotion_basis,
                    "promotion_history": promotion_history,
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

    def _append_promotion_history(
        self,
        *,
        history: List[Dict[str, Any]] | None,
        from_tier: str,
        to_tier: str,
        reason: str,
        basis: Dict[str, Any] | None,
        now_iso: str | None,
    ) -> List[Dict[str, Any]]:
        rows = list(history or [])
        rows.append(
            {
                "from_tier": self._tier(from_tier),
                "to_tier": self._tier(to_tier),
                "reason": reason,
                "basis": dict(basis or {}),
                "timestamp": str(now_iso or ""),
            }
        )
        return rows[-20:]

    def promote_local_to_workspace(
        self,
        promoted_id: str,
        *,
        reason: str = "local_to_workspace_threshold",
        now_iso: str | None = None,
    ) -> Dict[str, Any] | None:
        record = self.get(promoted_id)
        if not record:
            return None
        current_tier = self._tier(record.get("promotion_tier"))
        if current_tier in {self.TIER_WORKSPACE, self.TIER_GLOBAL}:
            return record

        confidence = float(record.get("confidence") or 0.0)
        evidence_count = len(list(record.get("evidence_ids") or []))
        has_source = bool((record.get("metadata") or {}).get("source_url") or (record.get("metadata") or {}).get("source_id"))
        contradiction_summary = self.get_contradiction_summary(promoted_id)
        open_conflicts = int(contradiction_summary.get("open_contradiction_count") or 0)
        eligible = confidence >= 0.6 and evidence_count >= 1 and has_source and open_conflicts == 0
        if not eligible:
            return record

        now = now_iso or ""
        basis = {
            "rule": reason,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "has_source": has_source,
            "open_contradiction_count": open_conflicts,
        }
        self._upsert_internal(
            promoted_id=record["promoted_id"],
            canonical_fact=record["canonical_fact"],
            evidence_ids=list(record.get("evidence_ids") or []),
            confidence=confidence,
            freshness_state=record["freshness_state"],
            metadata=dict(record.get("metadata") or {}),
            now_iso=now,
            promotion_state=record["promotion_state"],
            promotion_tier=self.TIER_WORKSPACE,
            origin_tier=record.get("origin_tier") or self.TIER_LOCAL,
            promoted_from_ids=sorted({*list(record.get("promoted_from_ids") or []), promoted_id}),
            promotion_basis=basis,
            promotion_history=self._append_promotion_history(
                history=list(record.get("promotion_history") or []),
                from_tier=current_tier,
                to_tier=self.TIER_WORKSPACE,
                reason=reason,
                basis=basis,
                now_iso=now,
            ),
            approval_reason=record.get("approval_reason") or "",
            approval_basis=dict(record.get("approval_basis") or {}),
            approved_at=record.get("approved_at"),
            freshness_lifecycle_state=record.get("freshness_lifecycle_state") or self.FRESHNESS_STALE,
            freshness_reason=record.get("freshness_reason") or "",
            freshness_policy=dict(record.get("freshness_policy") or {"ttl_hours": self.DEFAULT_TTL_HOURS}),
            last_validated_at=record.get("last_validated_at"),
            revalidation_requested_at=record.get("revalidation_requested_at"),
        )
        return self.get(promoted_id)

    def promote_workspace_to_global(
        self,
        promoted_id: str,
        *,
        reason: str = "workspace_to_global_strict_threshold",
        now_iso: str | None = None,
    ) -> Dict[str, Any] | None:
        record = self.get(promoted_id)
        if not record:
            return None
        current_tier = self._tier(record.get("promotion_tier"))
        if current_tier == self.TIER_GLOBAL:
            return record
        if current_tier != self.TIER_WORKSPACE:
            return record

        confidence = float(record.get("confidence") or 0.0)
        evidence_count = len(list(record.get("evidence_ids") or []))
        trusted = str(record.get("promotion_state") or "") == self.STATE_TRUSTED
        fresh = str(record.get("freshness_lifecycle_state") or "") == self.FRESHNESS_FRESH
        contradiction_summary = self.get_contradiction_summary(promoted_id)
        open_conflicts = int(contradiction_summary.get("open_contradiction_count") or 0)
        eligible = trusted and fresh and open_conflicts == 0 and confidence >= 0.8 and evidence_count >= 2
        if not eligible:
            return record

        now = now_iso or ""
        basis = {
            "rule": reason,
            "trusted": trusted,
            "fresh": fresh,
            "open_contradiction_count": open_conflicts,
            "confidence": confidence,
            "evidence_count": evidence_count,
        }
        self._upsert_internal(
            promoted_id=record["promoted_id"],
            canonical_fact=record["canonical_fact"],
            evidence_ids=list(record.get("evidence_ids") or []),
            confidence=confidence,
            freshness_state=record["freshness_state"],
            metadata=dict(record.get("metadata") or {}),
            now_iso=now,
            promotion_state=record["promotion_state"],
            promotion_tier=self.TIER_GLOBAL,
            origin_tier=record.get("origin_tier") or self.TIER_LOCAL,
            promoted_from_ids=sorted({*list(record.get("promoted_from_ids") or []), promoted_id}),
            promotion_basis=basis,
            promotion_history=self._append_promotion_history(
                history=list(record.get("promotion_history") or []),
                from_tier=current_tier,
                to_tier=self.TIER_GLOBAL,
                reason=reason,
                basis=basis,
                now_iso=now,
            ),
            approval_reason=record.get("approval_reason") or "",
            approval_basis=dict(record.get("approval_basis") or {}),
            approved_at=record.get("approved_at"),
            freshness_lifecycle_state=record.get("freshness_lifecycle_state") or self.FRESHNESS_STALE,
            freshness_reason=record.get("freshness_reason") or "",
            freshness_policy=dict(record.get("freshness_policy") or {"ttl_hours": self.DEFAULT_TTL_HOURS}),
            last_validated_at=record.get("last_validated_at"),
            revalidation_requested_at=record.get("revalidation_requested_at"),
        )
        return self.get(promoted_id)
