#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _post_json(url: str, payload: dict[str, Any], timeout_s: int = 20) -> tuple[int, dict[str, Any] | None, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw.strip() else {}
            return int(resp.status), body, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:
            parsed = None
        return int(e.code), parsed, raw
    except Exception as e:  # pragma: no cover - script-level failure path
        return 0, None, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run supplemental Phase-3 runtime readback proof (message-parts + structured export).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--outdir", default="", help="Optional artifact dir; defaults to forensics/<stamp>_phase3_readback")
    args = ap.parse_args()

    stamp = _stamp()
    outdir = Path(args.outdir) if args.outdir else Path("forensics") / f"{stamp}_phase3_readback"
    outdir.mkdir(parents=True, exist_ok=True)

    session_id = f"phase3-readback-{int(time.time())}"
    message_id = "m1"
    tool_call_id = f"tc-{int(time.time())}"
    base = args.base_url.rstrip("/")

    calls: list[dict[str, Any]] = []

    def call(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        status, body, raw = _post_json(f"{base}{path}", payload)
        calls.append({"path": path, "payload": payload, "status": status, "body": body, "raw": raw})
        return status, body

    checks = {
        "append_message_ok": False,
        "append_parts_ok": False,
        "structured_export_ok": False,
        "structured_export_has_parts": False,
        "structured_export_has_compaction": False,
        "recall_message_part_ok": False,
        "describe_tool_trace_ok": False,
        "expand_related_parts_ok": False,
    }

    st, body = call(
        "/session_append_message",
        {
            "session_id": session_id,
            "message_id": message_id,
            "sender": "assistant",
            "text": "runtime readback seed",
            "created_at": "2026-04-12T10:00:00Z",
        },
    )
    checks["append_message_ok"] = st == 200 and isinstance(body, dict) and body.get("status") in ("inserted", "updated")

    part_payloads = [
        {
            "session_id": session_id,
            "message_id": message_id,
            "part_index": 0,
            "text": "tool call started",
            "meta_json": {
                "part_type": "tool_call",
                "status": "started",
                "tool_name": "memory_search",
                "tool_call_id": tool_call_id,
            },
        },
        {
            "session_id": session_id,
            "message_id": message_id,
            "part_index": 1,
            "text": "tool call result",
            "meta_json": {
                "part_type": "tool_result",
                "status": "succeeded",
                "tool_name": "memory_search",
                "tool_call_id": tool_call_id,
                "retry_of_part_index": 0,
            },
        },
    ]
    part_ok = True
    for payload in part_payloads:
        st, body = call("/session_append_message_part", payload)
        part_ok = part_ok and st == 200 and isinstance(body, dict) and body.get("status") in ("inserted", "updated")
    checks["append_parts_ok"] = part_ok

    st, body = call("/session_structured_export", {"session_id": session_id})
    checks["structured_export_ok"] = st == 200 and isinstance(body, dict)
    if isinstance(body, dict):
        checks["structured_export_has_parts"] = int((body.get("part_stats") or {}).get("part_count") or 0) >= 2
        comp = body.get("compaction") or {}
        checks["structured_export_has_compaction"] = "segments" in comp and "quarantined" in comp

    st, body = call("/session_recall", {"session_id": session_id, "query": tool_call_id, "top_k": 10})
    ref: dict[str, Any] | None = None
    if st == 200 and isinstance(body, dict):
        refs = list(body.get("results") or [])
        for candidate in refs:
            if isinstance(candidate, dict) and candidate.get("item_type") == "message_part":
                ref = candidate
                break
    checks["recall_message_part_ok"] = ref is not None

    if ref is not None:
        st, body = call("/session_describe_item", {"ref": ref})
        checks["describe_tool_trace_ok"] = (
            st == 200
            and isinstance(body, dict)
            and body.get("item_type") == "message_part"
            and body.get("tool_call_id") == tool_call_id
            and body.get("tool_name") == "memory_search"
        )
        st, body = call("/session_expand_item", {"ref": ref})
        expanded = list((body or {}).get("expanded") or []) if isinstance(body, dict) else []
        checks["expand_related_parts_ok"] = st == 200 and any(
            isinstance(item, dict) and item.get("item_type") == "message_part" for item in expanded
        )

    passed = all(checks.values())
    summary = {
        "schemaVersion": "phase3_supplemental_readback.v1",
        "startedAt": stamp,
        "baseUrl": base,
        "sessionId": session_id,
        "toolCallId": tool_call_id,
        "checks": checks,
        "passed": passed,
        "artifactDir": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
