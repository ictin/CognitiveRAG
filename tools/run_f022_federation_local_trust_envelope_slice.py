#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.federation.local_trust_envelope import LocalFederationEnvelopeStore
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _seed_reasoning_db(workdir: Path) -> None:
    db_path = workdir / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
        )
        db.execute(
            "INSERT OR REPLACE INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "p_f022",
                "postgres migration rollback timeout",
                "[]",
                "Use staged rollback checklist with timeout fallback and verification.",
                0.84,
                "[]",
                "workflow_pattern",
                "postgres migration rollback timeout workflow",
                "current",
            ),
        )
        db.commit()


def _seed_corpus_db(workdir: Path) -> None:
    db_path = workdir / "context_items.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS context_items (item_id TEXT, session_id TEXT, type TEXT, payload_json TEXT, created_at TEXT)"
        )
        payload = {
            "summary": "Postgres migration rollback checklist and timeout fallback strategy.",
            "file_path": "books/postgres_ops.md",
        }
        db.execute(
            "INSERT INTO context_items VALUES (?, ?, ?, ?, ?)",
            ("f022-corpus-1", "f022-session", "corpus_chunk", json.dumps(payload), "2026-05-05T12:00:00Z"),
        )
        db.commit()


def _run_route(workdir: str):
    clear_routing_caches()
    plan, hits = route_and_retrieve(
        query="postgres migration rollback timeout",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="f022-fallback-proof",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=workdir,
        top_k_per_lane=6,
    )
    return [l.value for l in plan.lanes], [
        {
            "id": h.id,
            "lane": h.lane.value,
            "memory_type": h.memory_type.value,
            "provenance": dict(h.provenance or {}),
        }
        for h in hits
    ]


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f022_federation_local_trust_envelope_slice"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    _seed_reasoning_db(workdir)
    _seed_corpus_db(workdir)

    store = LocalFederationEnvelopeStore(workdir / "federation_packets.sqlite3")
    exported = store.export_packet(
        source_install_id="local-install-a",
        payload_class="reasoning_pattern",
        source_object_ids=["promoted:p_f022", "webpromoted:wp_pg"],
        provenance_refs=[
            {"type": "reasoning_pattern", "id": "p_f022", "uri": "memory://reasoning/p_f022"},
            {"type": "web_promoted", "id": "wp_pg", "uri": "memory://web/wp_pg"},
        ],
        payload={"summary": "Rollback strategy with verification guardrails", "confidence": 0.82},
        freshness_lifecycle_state="revalidation_required",
        metadata={"origin_note": "local_export_only"},
    )
    imported = store.import_packet(exported)
    readback = store.read_packet(imported["packet_id"])

    (outdir / "federation_packet_export_artifact.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    (outdir / "federation_packet_import_readback_artifact.json").write_text(
        json.dumps({"imported": imported, "readback": readback}, indent=2), encoding="utf-8"
    )
    (outdir / "trust_envelope_provenance_artifact.json").write_text(
        json.dumps(
            {
                "packet_id": readback.get("packet_id"),
                "trust_status": readback.get("trust_status"),
                "approval_status": readback.get("approval_status"),
                "freshness_lifecycle_state": readback.get("freshness_lifecycle_state"),
                "provenance_refs": list(readback.get("provenance_refs") or []),
                "source_object_ids": list(readback.get("source_object_ids") or []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "quarantine_non_authoritative_import_artifact.json").write_text(
        json.dumps(
            {
                "import_state": readback.get("import_state"),
                "authoritative": readback.get("authoritative"),
                "trust_status": readback.get("trust_status"),
                "approval_status": readback.get("approval_status"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lanes_enabled, hits_enabled = _run_route(str(workdir))
    os.environ["CRAG_DISABLE_FEDERATION_HELPER"] = "1"
    lanes_disabled, hits_disabled = _run_route(str(workdir))
    os.environ.pop("CRAG_DISABLE_FEDERATION_HELPER", None)

    (outdir / "federation_disabled_fallback_artifact.json").write_text(
        json.dumps(
            {
                "lanes_enabled": lanes_enabled,
                "lanes_disabled": lanes_disabled,
                "selector_authority_preserved": lanes_enabled == lanes_disabled,
                "enabled_hit_count": len(hits_enabled),
                "disabled_hit_count": len(hits_disabled),
                "retrieval_provenance_contains_federation": any("federation" in (h.get("provenance") or {}) for h in hits_enabled),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "explanation_trace_non_authoritative_artifact.json").write_text(
        json.dumps(
            {
                "federation_packet": {
                    "packet_id": readback.get("packet_id"),
                    "state": readback.get("import_state"),
                    "trust_status": readback.get("trust_status"),
                    "approval_status": readback.get("approval_status"),
                    "authoritative": bool(readback.get("authoritative")),
                    "label": "non_authoritative_quarantine",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    checks = {
        "stable_packet_id_present": str(readback.get("packet_id") or "").startswith("fedpkt:"),
        "provenance_preserved": bool(readback.get("provenance_refs")) and bool(readback.get("source_object_ids")),
        "quarantined_non_authoritative_import": (
            readback.get("import_state") == LocalFederationEnvelopeStore.IMPORT_STATE_QUARANTINED
            and (bool(readback.get("authoritative")) is False)
        ),
        "no_auto_trust_or_promotion": (
            readback.get("trust_status") == LocalFederationEnvelopeStore.TRUST_UNTRUSTED
            and readback.get("approval_status") == LocalFederationEnvelopeStore.APPROVAL_UNREVIEWED
        ),
        "federation_disabled_fallback_works": lanes_enabled == lanes_disabled,
        "retrieval_not_authority_impacted": not any("federation" in (h.get("provenance") or {}) for h in hits_enabled),
    }
    summary = {
        "schemaVersion": "f022_federation_local_trust_envelope_slice.v1",
        "artifactDir": str(outdir),
        "checks": checks,
        "traceability": {
            "requested_feature_id": "F-022",
            "mapped_feature_id": "F-023",
            "related_features": ["F-006", "F-007", "F-009", "F-015"],
            "invariants": ["INV-009", "INV-016"],
            "workflows": ["WF-005"],
        },
    }
    summary["passed"] = all(checks.values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
