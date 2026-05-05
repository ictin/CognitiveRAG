from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_packet_id(
    *,
    source_install_id: str,
    payload_class: str,
    source_object_ids: List[str],
    provenance_refs: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> str:
    seed = {
        "source_install_id": str(source_install_id or "").strip(),
        "payload_class": str(payload_class or "").strip().lower(),
        "source_object_ids": sorted(str(x).strip() for x in (source_object_ids or []) if str(x).strip()),
        "provenance_refs": sorted(
            [
                {
                    "type": str(r.get("type") or "").strip().lower(),
                    "id": str(r.get("id") or "").strip(),
                    "uri": str(r.get("uri") or "").strip(),
                }
                for r in (provenance_refs or [])
            ],
            key=lambda r: (r["type"], r["id"], r["uri"]),
        ),
        "payload_digest": hashlib.sha1(_canonical_json(payload or {}).encode("utf-8")).hexdigest(),
    }
    return "fedpkt:" + hashlib.sha1(_canonical_json(seed).encode("utf-8")).hexdigest()[:20]


class LocalFederationEnvelopeStore:
    IMPORT_STATE_QUARANTINED = "quarantined"
    TRUST_UNTRUSTED = "untrusted"
    APPROVAL_UNREVIEWED = "unreviewed"

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
                CREATE TABLE IF NOT EXISTS federation_packets (
                    packet_id TEXT PRIMARY KEY,
                    source_install_id TEXT NOT NULL,
                    payload_class TEXT NOT NULL,
                    source_object_ids_json TEXT NOT NULL,
                    provenance_refs_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    trust_status TEXT NOT NULL,
                    approval_status TEXT NOT NULL,
                    freshness_lifecycle_state TEXT NOT NULL,
                    import_state TEXT NOT NULL,
                    authoritative INTEGER NOT NULL DEFAULT 0,
                    exported_at TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fed_import_state ON federation_packets(import_state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fed_payload_class ON federation_packets(payload_class)")

    def export_packet(
        self,
        *,
        source_install_id: str,
        payload_class: str,
        source_object_ids: List[str],
        provenance_refs: List[Dict[str, Any]],
        payload: Dict[str, Any],
        freshness_lifecycle_state: str = "revalidation_required",
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        exported_at = _now_iso()
        packet_id = stable_packet_id(
            source_install_id=source_install_id,
            payload_class=payload_class,
            source_object_ids=source_object_ids,
            provenance_refs=provenance_refs,
            payload=payload,
        )
        return {
            "packet_id": packet_id,
            "source_install_id": str(source_install_id or "").strip(),
            "created_at": exported_at,
            "payload_class": str(payload_class or "").strip().lower(),
            "source_object_ids": sorted(str(x).strip() for x in (source_object_ids or []) if str(x).strip()),
            "provenance_refs": list(provenance_refs or []),
            "payload": dict(payload or {}),
            "trust_status": self.TRUST_UNTRUSTED,
            "approval_status": self.APPROVAL_UNREVIEWED,
            "freshness_lifecycle_state": str(freshness_lifecycle_state or "revalidation_required").strip().lower(),
            "import_state": self.IMPORT_STATE_QUARANTINED,
            "authoritative": False,
            "metadata": dict(metadata or {}),
        }

    def import_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        imported_at = _now_iso()
        normalized = {
            "packet_id": str(packet.get("packet_id") or "").strip(),
            "source_install_id": str(packet.get("source_install_id") or "").strip(),
            "payload_class": str(packet.get("payload_class") or "").strip().lower(),
            "source_object_ids": sorted(str(x).strip() for x in (packet.get("source_object_ids") or []) if str(x).strip()),
            "provenance_refs": list(packet.get("provenance_refs") or []),
            "payload": dict(packet.get("payload") or {}),
            "trust_status": self.TRUST_UNTRUSTED,
            "approval_status": self.APPROVAL_UNREVIEWED,
            "freshness_lifecycle_state": str(packet.get("freshness_lifecycle_state") or "revalidation_required").strip().lower(),
            "import_state": self.IMPORT_STATE_QUARANTINED,
            "authoritative": False,
            "created_at": str(packet.get("created_at") or imported_at),
            "metadata": dict(packet.get("metadata") or {}),
        }
        if not normalized["packet_id"]:
            normalized["packet_id"] = stable_packet_id(
                source_install_id=normalized["source_install_id"],
                payload_class=normalized["payload_class"],
                source_object_ids=normalized["source_object_ids"],
                provenance_refs=normalized["provenance_refs"],
                payload=normalized["payload"],
            )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO federation_packets(
                    packet_id, source_install_id, payload_class, source_object_ids_json, provenance_refs_json, payload_json,
                    trust_status, approval_status, freshness_lifecycle_state, import_state, authoritative,
                    exported_at, imported_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(packet_id) DO UPDATE SET
                    source_install_id=excluded.source_install_id,
                    payload_class=excluded.payload_class,
                    source_object_ids_json=excluded.source_object_ids_json,
                    provenance_refs_json=excluded.provenance_refs_json,
                    payload_json=excluded.payload_json,
                    trust_status=excluded.trust_status,
                    approval_status=excluded.approval_status,
                    freshness_lifecycle_state=excluded.freshness_lifecycle_state,
                    import_state=excluded.import_state,
                    authoritative=excluded.authoritative,
                    exported_at=excluded.exported_at,
                    imported_at=excluded.imported_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    normalized["packet_id"],
                    normalized["source_install_id"],
                    normalized["payload_class"],
                    json.dumps(normalized["source_object_ids"]),
                    json.dumps(normalized["provenance_refs"]),
                    json.dumps(normalized["payload"]),
                    normalized["trust_status"],
                    normalized["approval_status"],
                    normalized["freshness_lifecycle_state"],
                    normalized["import_state"],
                    0,
                    normalized["created_at"],
                    imported_at,
                    json.dumps(normalized["metadata"]),
                ),
            )
            conn.commit()
        return self.read_packet(normalized["packet_id"])

    def read_packet(self, packet_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM federation_packets WHERE packet_id = ?", (str(packet_id),)).fetchone()
        if row is None:
            return {}
        return {
            "packet_id": row["packet_id"],
            "source_install_id": row["source_install_id"],
            "payload_class": row["payload_class"],
            "source_object_ids": json.loads(row["source_object_ids_json"] or "[]"),
            "provenance_refs": json.loads(row["provenance_refs_json"] or "[]"),
            "payload": json.loads(row["payload_json"] or "{}"),
            "trust_status": row["trust_status"],
            "approval_status": row["approval_status"],
            "freshness_lifecycle_state": row["freshness_lifecycle_state"],
            "import_state": row["import_state"],
            "authoritative": bool(int(row["authoritative"] or 0)),
            "exported_at": row["exported_at"],
            "imported_at": row["imported_at"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    def list_packets(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT packet_id FROM federation_packets ORDER BY imported_at ASC, packet_id ASC").fetchall()
        return [self.read_packet(str(r["packet_id"])) for r in rows]
