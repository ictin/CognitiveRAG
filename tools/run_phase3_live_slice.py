#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _now_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_gateway_json(raw: str) -> dict[str, Any]:
    idx = raw.find("{")
    if idx < 0:
        raise ValueError("no JSON payload in gateway output")
    return json.loads(raw[idx:])


def _gateway_call(method: str, params: dict[str, Any], timeout_s: int = 50, retries: int = 2) -> tuple[dict[str, Any] | None, dict[str, Any]]:
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
            rec = {
                "method": method,
                "params": params,
                "attempt": attempt,
                "rc": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            if proc.returncode != 0:
                last_err = (proc.stderr or proc.stdout or "").strip()
                if "1006" in last_err or "timeout" in last_err.lower() or "gateway closed" in last_err.lower():
                    time.sleep(1.5 * attempt)
                    continue
                return None, rec
            try:
                payload = _parse_gateway_json(proc.stdout)
                rec["ok"] = True
                return payload, rec
            except Exception as exc:  # pragma: no cover - defensive
                rec["ok"] = False
                rec["parse_error"] = str(exc)
                return None, rec
        except subprocess.TimeoutExpired as exc:
            last_err = f"timeout_expired: {exc}"
            if attempt < retries:
                time.sleep(1.5 * attempt)
                continue
    return None, {
        "method": method,
        "params": params,
        "ok": False,
        "rc": -1,
        "stderr": last_err or "gateway call failed",
    }


def _text_of_message(msg: dict[str, Any]) -> str:
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


def _wait_new_assistant(
    session_key: str,
    previous_assistant_count: int,
    wait_s: int = 75,
) -> tuple[dict[str, Any] | None, dict[str, Any], int]:
    deadline = time.time() + wait_s
    last_obj: dict[str, Any] = {}
    last_assistant: dict[str, Any] | None = None
    last_assistant_count = previous_assistant_count
    while time.time() < deadline:
        obj, rec = _gateway_call("sessions.get", {"key": session_key}, timeout_s=35, retries=2)
        if obj is None:
            time.sleep(1.2)
            continue
        last_obj = obj
        msgs = obj.get("messages") or []
        assistants = [m for m in msgs if m.get("role") == "assistant"]
        if len(assistants) > previous_assistant_count:
            candidate = assistants[-1]
            candidate_text = _text_of_message(candidate).strip()
            last_assistant = candidate
            last_assistant_count = len(assistants)
            # "NO_REPLY" can be an intermediate degraded output in flaky runtime paths.
            # Keep polling within the same step if it appears, so we capture a real answer when available.
            if candidate_text and candidate_text.upper() != "NO_REPLY":
                return candidate, rec, len(assistants)
        time.sleep(1.2)
    if last_assistant is not None:
        return last_assistant, {"method": "sessions.get", "timeout_waiting_non_no_reply_assistant": True}, last_assistant_count
    return None, {"method": "sessions.get", "timeout_waiting_assistant": True}, previous_assistant_count


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a stable Phase 3 live validation slice against OpenClaw gateway.")
    ap.add_argument("--outdir", default="", help="Optional output dir. Default: forensics/<stamp>_phase3_live_slice")
    ap.add_argument("--label", default="Phase3 live slice")
    ap.add_argument("--model", default="gpt-5-mini", help="Gateway model to lock for deterministic runtime validation.")
    args = ap.parse_args()

    stamp = _now_stamp()
    outdir = Path(args.outdir) if args.outdir else Path("forensics") / f"{stamp}_phase3_live_slice"
    outdir.mkdir(parents=True, exist_ok=True)

    session_key = f"agent:main:phase3-live-slice-{int(time.time())}"

    calls: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []

    label = f"{args.label} {stamp}"
    create_params = {"key": session_key, "label": label}
    if args.model:
        create_params["model"] = args.model
    created, rec = _gateway_call("sessions.create", create_params, timeout_s=35, retries=2)
    calls.append(rec)
    if created is None:
        summary = {
            "passed": False,
            "reason": "session_create_failed",
            "sessionKey": session_key,
            "sessionLabel": label,
            "model": args.model,
            "artifactDir": str(outdir),
            "calls": len(calls),
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    steps = [
        ("status", "/crag_status"),
        ("explain", "/crag_explain_memory"),
        (
            "seed_task",
            "Task scenario seed: objective close phase3 lossless memory; blockers mirror-heavy task-state and unstable heavy benchmark; next steps align runtime path, fix quote-span, fix task-state grounding; acceptance criteria exact quote span and session-grounded task-state answers.",
        ),
        ("ask_task_state", "What is the current task state in this session?"),
        ("ask_blockers", "What are the current blockers?"),
        ("ask_next_steps", "What are the next steps?"),
        ("seed_quote", "Quote seed: EXACT-SPAN-DELTA-88219 appears in this exact sentence and should be quoted exactly later."),
        ("ask_quote", "What did we say earlier about EXACT-SPAN-DELTA-88219? Quote the earlier span exactly."),
        (
            "long_continuity",
            "Long continuity check: summarize objective, blockers, and next steps in 3 bullets with source grounding.",
        ),
    ]

    assistant_count = 0
    for step_id, prompt in steps:
        _, rec = _gateway_call("sessions.send", {"key": session_key, "message": prompt}, timeout_s=55, retries=2)
        rec["step"] = step_id
        calls.append(rec)
        assistant_msg, poll_rec, assistant_count = _wait_new_assistant(session_key, assistant_count, wait_s=80)
        calls.append({"step": step_id, "poll": poll_rec})

        assistant_text = _text_of_message(assistant_msg) if assistant_msg else ""
        transcript.append(
            {
                "step": step_id,
                "prompt": prompt,
                "assistantReceived": bool(assistant_text),
                "assistant": assistant_text,
            }
        )

    sess_obj, rec = _gateway_call("sessions.get", {"key": session_key}, timeout_s=45, retries=2)
    calls.append(rec)

    checks = {
        "status_has_runtime_path": False,
        "explain_has_architecture": False,
        "task_state_answer_present": False,
        "blockers_answer_present": False,
        "next_steps_answer_present": False,
        "quote_contains_seed_token": False,
        "continuity_answer_present": False,
    }

    for row in transcript:
        txt = (row.get("assistant") or "").lower()
        if row["step"] == "status":
            checks["status_has_runtime_path"] = "runtime entry path:" in txt
        elif row["step"] == "explain":
            checks["explain_has_architecture"] = "cognitiverag memory architecture" in txt
        elif row["step"] == "ask_task_state":
            checks["task_state_answer_present"] = bool(txt)
        elif row["step"] == "ask_blockers":
            checks["blockers_answer_present"] = bool(txt)
        elif row["step"] == "ask_next_steps":
            checks["next_steps_answer_present"] = bool(txt)
        elif row["step"] == "ask_quote":
            checks["quote_contains_seed_token"] = "exact-span-delta-88219" in txt
        elif row["step"] == "long_continuity":
            checks["continuity_answer_present"] = bool(txt)

    passed = all(checks.values())
    summary = {
        "schemaVersion": "phase3_live_slice.v1",
        "startedAt": stamp,
        "sessionKey": session_key,
        "sessionLabel": label,
        "model": args.model,
        "artifactDir": str(outdir),
        "checks": checks,
        "passed": passed,
        "notes": [
            "This is the stable replacement live slice for Phase 3 when heavy benchmark path is unstable.",
            "Failure here indicates runtime trust is still insufficient for closure.",
        ],
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
    (outdir / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    (outdir / "session_get.json").write_text(json.dumps(sess_obj or {}, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
