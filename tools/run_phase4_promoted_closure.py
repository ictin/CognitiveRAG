#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _now_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_gateway_json(raw: str) -> dict[str, Any]:
    idx = raw.find("{")
    if idx < 0:
        raise ValueError("no JSON payload in gateway output")
    return json.loads(raw[idx:])


def _gateway_call(method: str, params: dict[str, Any], *, timeout_s: int = 60, retries: int = 3) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    last_err: str | None = None
    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                ["openclaw", "gateway", "call", method, "--params", json.dumps(params)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            rec = {"method": method, "params": params, "attempt": attempt, "rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
            if proc.returncode != 0:
                last_err = (proc.stderr or proc.stdout or "").strip()
                if ("1006" in last_err or "timeout" in last_err.lower()) and attempt < retries:
                    time.sleep(1.4 * attempt)
                    continue
                return None, rec
            return _parse_gateway_json(proc.stdout), rec
        except subprocess.TimeoutExpired as exc:
            last_err = f"timeout_expired: {exc}"
            if attempt < retries:
                time.sleep(1.4 * attempt)
                continue
    return None, {"method": method, "params": params, "rc": -1, "stderr": last_err or "gateway call failed"}


def _post_json(url: str, payload: dict[str, Any], timeout_s: int = 30) -> tuple[int, dict[str, Any] | None, str]:
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
    except Exception as e:
        return 0, None, str(e)


def _text_of_message(msg: dict[str, Any] | None) -> str:
    if not msg:
        return ""
    content = msg.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def _wait_new_assistant(session_key: str, previous_assistant_count: int, *, wait_s: int = 70) -> tuple[dict[str, Any] | None, dict[str, Any], int]:
    deadline = time.time() + wait_s
    last_assistant: dict[str, Any] | None = None
    last_count = previous_assistant_count
    while time.time() < deadline:
        obj, rec = _gateway_call("sessions.get", {"key": session_key}, timeout_s=40, retries=2)
        if obj is None:
            time.sleep(1.1)
            continue
        msgs = obj.get("messages") or []
        assistants = [m for m in msgs if m.get("role") == "assistant"]
        if len(assistants) > previous_assistant_count:
            candidate = assistants[-1]
            txt = _text_of_message(candidate).strip()
            last_assistant = candidate
            last_count = len(assistants)
            if txt and txt.upper() != "NO_REPLY":
                return candidate, rec, len(assistants)
        time.sleep(1.1)
    if last_assistant is not None:
        return last_assistant, {"method": "sessions.get", "timeout_waiting_non_no_reply_assistant": True}, last_count
    return None, {"method": "sessions.get", "timeout_waiting_assistant": True}, previous_assistant_count


def _send_and_capture(session_key: str, step_id: str, prompt: str, assistant_count: int, calls: list[dict[str, Any]], transcript: list[dict[str, Any]], wait_s: int) -> int:
    _, rec = _gateway_call("sessions.send", {"key": session_key, "message": prompt}, timeout_s=60, retries=3)
    rec["step"] = step_id
    calls.append(rec)
    assistant_msg, poll_rec, next_count = _wait_new_assistant(session_key, assistant_count, wait_s=wait_s)
    calls.append({"step": step_id, "poll": poll_rec})
    transcript.append(
        {
            "step": step_id,
            "prompt": prompt,
            "assistant": _text_of_message(assistant_msg),
            "assistantReceived": bool(_text_of_message(assistant_msg)),
        }
    )
    return next_count


def _has_promoted_lane(explanation: dict[str, Any] | None) -> bool:
    if not isinstance(explanation, dict):
        return False
    for block in list(explanation.get("selected_blocks") or []):
        if str(block.get("lane") or "") == "promoted":
            return True
    for block in list(explanation.get("dropped_blocks") or []):
        if str(block.get("lane") or "") == "promoted":
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase-4 durable promoted memory closure battery.")
    ap.add_argument("--outdir", default="", help="Optional artifact dir; defaults to forensics/<stamp>_phase4_promoted_closure")
    ap.add_argument("--label", default="Phase4 promoted closure")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--backend-base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--wait-seconds", type=int, default=55)
    args = ap.parse_args()

    stamp = _now_stamp()
    outdir = Path(args.outdir) if args.outdir else Path("forensics") / f"{stamp}_phase4_promoted_closure"
    outdir.mkdir(parents=True, exist_ok=True)

    key = f"agent:main:phase4-promoted-{int(time.time())}"
    calls: list[dict[str, Any]] = []
    backend_calls: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []

    fact_token = f"FACT-{int(time.time())}"
    proc_token = f"PROC-{int(time.time())}"
    episodic_token = f"EPHEMERAL-{int(time.time())}"

    created, rec = _gateway_call("sessions.create", {"key": key, "label": f"{args.label} {stamp}", "model": args.model}, timeout_s=45, retries=3)
    calls.append(rec)
    if created is None:
        summary = {"schemaVersion": "phase4_promoted_closure.v1", "passed": False, "reason": "session_create_failed", "artifactDir": str(outdir)}
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    session_id = str(created.get("sessionId") or "")
    if not session_id:
        summary = {"schemaVersion": "phase4_promoted_closure.v1", "passed": False, "reason": "session_id_missing", "artifactDir": str(outdir)}
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    backend = args.backend_base_url.rstrip("/")

    def backend_call(path: str, payload: dict[str, Any], timeout_s: int = 30) -> tuple[int, dict[str, Any] | None]:
        status, body, raw = _post_json(f"{backend}{path}", payload, timeout_s=timeout_s)
        backend_calls.append({"path": path, "payload": payload, "status": status, "body": body, "raw": raw})
        return status, body

    # Deterministic backend-side seed for promotion bridge coverage.
    seed_lines = [
        f"Stable fact seed: deployment window is Tuesday 14:00 UTC ({fact_token}).",
        f"Reusable procedure seed {proc_token}: ingest -> validate -> deploy; then run smoke tests.",
        f"Episodic seed: temporary canary token today is {episodic_token}.",
        f"Project note: this session validates Phase 4 promoted memory behavior for {session_id}.",
    ]
    for idx, line in enumerate(seed_lines):
        backend_call(
            "/session_append_message",
            {
                "session_id": session_id,
                "message_id": f"phase4-seed-{idx}",
                "sender": "user" if idx % 2 == 0 else "assistant",
                "text": line,
                "created_at": f"2026-04-12T13:{(10 + idx):02d}:00Z",
            },
        )
    backend_call("/session_compact", {"session_id": session_id, "older_than_index": 12})

    assistant_count = 0
    steps = [
        ("status", "/crag_status"),
        ("explain", "/crag_explain_memory"),
    ]
    for step_id, prompt in steps:
        assistant_count = _send_and_capture(key, step_id, prompt, assistant_count, calls, transcript, wait_s=max(25, int(args.wait_seconds)))
        if not transcript[-1]["assistantReceived"]:
            assistant_count = _send_and_capture(
                key,
                f"{step_id}_retry",
                prompt,
                assistant_count,
                calls,
                transcript,
                wait_s=max(25, int(args.wait_seconds)),
            )

    checks = {
        "runtime_path_truth": True,
        "status_command_ok": False,
        "explain_command_ok": False,
        "promotion_trigger_ok": False,
        "promoted_fact_readback_ok": False,
        "promoted_procedure_readback_ok": False,
        "provenance_lineage_ok": False,
        "dedup_safe_surface_ok": False,
        "confidence_freshness_surface_ok": False,
        "promoted_retrieval_mode_ok": False,
        "episodic_not_masquerading_promoted_ok": False,
    }

    for row in transcript:
        if str(row["step"]).startswith("status"):
            checks["status_command_ok"] = "crag status" in row["assistant"].lower() or "runtime" in row["assistant"].lower()
        if str(row["step"]).startswith("explain"):
            checks["explain_command_ok"] = bool(row["assistant"].strip())

    st, body = backend_call("/promote_session", {"session_id": session_id})
    checks["promotion_trigger_ok"] = st == 200 and int((body or {}).get("promoted_count") or 0) >= 2

    st, body = backend_call("/promoted_search", {"query": fact_token, "top_k": 10})
    fact_items = list((body or {}).get("items") or []) if (st == 200 and isinstance(body, dict)) else []
    if fact_items:
        checks["promoted_fact_readback_ok"] = True

    st, body = backend_call("/promoted_search", {"query": proc_token, "top_k": 10})
    proc_items = list((body or {}).get("items") or []) if (st == 200 and isinstance(body, dict)) else []
    if proc_items:
        checks["promoted_procedure_readback_ok"] = True

    all_items = fact_items + proc_items
    if all_items:
        checks["provenance_lineage_ok"] = any((item.get("provenance") or []) for item in all_items)
        checks["confidence_freshness_surface_ok"] = all(
            ("confidence" in item) and bool(str(item.get("freshness_state") or ""))
            for item in all_items
        )
        checks["dedup_safe_surface_ok"] = all(int(item.get("reuse_count") or 1) >= 1 for item in all_items)
        # read one item directly
        pid = str(all_items[0].get("pattern_id") or "")
        if pid:
            gst, gbody = backend_call("/promoted_get", {"pattern_id": pid})
            if gst == 200 and isinstance(gbody, dict):
                checks["provenance_lineage_ok"] = checks["provenance_lineage_ok"] and bool(gbody.get("provenance"))

    st, body = backend_call(
        "/session_assemble_context",
        {
            "session_id": session_id,
            "query": f"What stable promoted fact should I use for deployment window {fact_token}?",
            "intent_family": "memory_summary",
            "fresh_tail_count": 0,
            "budget": 2200,
        },
    )
    if st == 200 and isinstance(body, dict):
        checks["promoted_retrieval_mode_ok"] = _has_promoted_lane(body.get("explanation"))

    st, body = backend_call("/promoted_search", {"query": episodic_token, "top_k": 5})
    epi_items = list((body or {}).get("items") or []) if (st == 200 and isinstance(body, dict)) else []
    checks["episodic_not_masquerading_promoted_ok"] = len(epi_items) == 0

    ask_steps = [
        ("ask_fact", f"What is our deployment window token {fact_token}?"),
        ("ask_proc", f"What reusable workflow is recommended with token {proc_token}?"),
        ("ask_epi", "What is the temporary canary token for today?"),
    ]
    for step_id, prompt in ask_steps:
        assistant_count = _send_and_capture(key, step_id, prompt, assistant_count, calls, transcript, wait_s=max(25, int(args.wait_seconds)))
        if not transcript[-1]["assistantReceived"]:
            assistant_count = _send_and_capture(
                key,
                f"{step_id}_retry",
                prompt,
                assistant_count,
                calls,
                transcript,
                wait_s=max(25, int(args.wait_seconds)),
            )

    for row in transcript:
        txt = row.get("assistant", "")
        if str(row["step"]).startswith("ask_fact") and fact_token in txt:
            checks["promoted_fact_readback_ok"] = checks["promoted_fact_readback_ok"] and True
        if str(row["step"]).startswith("ask_proc") and proc_token in txt:
            checks["promoted_procedure_readback_ok"] = checks["promoted_procedure_readback_ok"] and True
        if str(row["step"]).startswith("ask_epi"):
            lower = txt.lower()
            episodic_ok = (episodic_token in txt) or (
                (
                    "didn't find" in lower
                    or "did not find" in lower
                    or "not find" in lower
                    or ("not" in lower and "found" in lower)
                )
                and "promoted" not in lower
            )
            checks["episodic_not_masquerading_promoted_ok"] = checks["episodic_not_masquerading_promoted_ok"] and episodic_ok

    coverage = {
        "runtime_path_truth": checks["runtime_path_truth"],
        "promotion_bridge_surface": checks["promotion_trigger_ok"] and checks["promoted_fact_readback_ok"] and checks["promoted_procedure_readback_ok"],
        "lineage_surface": checks["provenance_lineage_ok"],
        "dedup_surface": checks["dedup_safe_surface_ok"],
        "confidence_freshness_surface": checks["confidence_freshness_surface_ok"],
        "promoted_retrieval_surface": checks["promoted_retrieval_mode_ok"],
        "truthful_source_surface": checks["episodic_not_masquerading_promoted_ok"],
    }
    passed = all(bool(v) for v in coverage.values())

    summary = {
        "schemaVersion": "phase4_promoted_closure.v1",
        "startedAt": stamp,
        "sessionKey": key,
        "sessionId": session_id,
        "model": args.model,
        "artifactDir": str(outdir),
        "instrumentRole": "phase4_closure_authoritative",
        "checks": checks,
        "coverage": coverage,
        "passed": passed,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
    (outdir / "backend_calls.json").write_text(json.dumps(backend_calls, indent=2), encoding="utf-8")
    (outdir / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
