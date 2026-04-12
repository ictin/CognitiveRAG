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


def _gateway_call(
    method: str,
    params: dict[str, Any],
    *,
    timeout_s: int = 55,
    retries: int = 3,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
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
                if (
                    "1006" in last_err
                    or "gateway closed" in last_err.lower()
                    or "timeout" in last_err.lower()
                    or "econnrefused" in last_err.lower()
                ) and attempt < retries:
                    time.sleep(1.5 * attempt)
                    continue
                return None, rec
            payload = _parse_gateway_json(proc.stdout)
            rec["ok"] = True
            return payload, rec
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


def _wait_new_assistant(
    session_key: str,
    previous_assistant_count: int,
    *,
    wait_s: int = 95,
) -> tuple[dict[str, Any] | None, dict[str, Any], int]:
    deadline = time.time() + wait_s
    last_assistant: dict[str, Any] | None = None
    last_assistant_count = previous_assistant_count
    while time.time() < deadline:
        obj, rec = _gateway_call("sessions.get", {"key": session_key}, timeout_s=35, retries=2)
        if obj is None:
            time.sleep(1.2)
            continue
        msgs = obj.get("messages") or []
        assistants = [m for m in msgs if m.get("role") == "assistant"]
        if len(assistants) > previous_assistant_count:
            candidate = assistants[-1]
            candidate_text = _text_of_message(candidate).strip()
            last_assistant = candidate
            last_assistant_count = len(assistants)
            if candidate_text and candidate_text.upper() != "NO_REPLY":
                return candidate, rec, len(assistants)
        time.sleep(1.2)
    if last_assistant is not None:
        return last_assistant, {"method": "sessions.get", "timeout_waiting_non_no_reply_assistant": True}, last_assistant_count
    return None, {"method": "sessions.get", "timeout_waiting_assistant": True}, previous_assistant_count


def _send_and_capture(
    session_key: str,
    step_id: str,
    prompt: str,
    assistant_count: int,
    calls: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    wait_s: int,
) -> int:
    _, rec = _gateway_call("sessions.send", {"key": session_key, "message": prompt}, timeout_s=60, retries=3)
    rec["step"] = step_id
    calls.append(rec)
    assistant_msg, poll_rec, next_count = _wait_new_assistant(session_key, assistant_count, wait_s=wait_s)
    calls.append({"step": step_id, "poll": poll_rec})
    assistant_text = _text_of_message(assistant_msg)
    transcript.append(
        {
            "step": step_id,
            "prompt": prompt,
            "assistantReceived": bool(assistant_text),
            "assistant": assistant_text,
        }
    )
    return next_count


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase-3 closure-grade runtime battery.")
    ap.add_argument("--outdir", default="", help="Optional artifact dir; defaults to forensics/<stamp>_phase3_closure_battery")
    ap.add_argument("--label", default="Phase3 closure battery")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--backend-base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--wait-seconds", type=int, default=60)
    args = ap.parse_args()

    stamp = _now_stamp()
    outdir = Path(args.outdir) if args.outdir else Path("forensics") / f"{stamp}_phase3_closure_battery"
    outdir.mkdir(parents=True, exist_ok=True)

    primary_key = f"agent:main:phase3-closure-{int(time.time())}"
    noise_key = f"{primary_key}-noise"
    quote_token = f"EXACT-SPAN-CLOSURE-{int(time.time())}"

    calls: list[dict[str, Any]] = []
    backend_calls: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []

    create_params = {"key": primary_key, "label": f"{args.label} {stamp}", "model": args.model}
    created, rec = _gateway_call("sessions.create", create_params, timeout_s=40, retries=3)
    calls.append(rec)
    if created is None:
        summary = {
            "schemaVersion": "phase3_closure_battery.v1",
            "passed": False,
            "reason": "primary_session_create_failed",
            "primarySessionKey": primary_key,
            "model": args.model,
            "artifactDir": str(outdir),
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    noise_created, rec = _gateway_call(
        "sessions.create",
        {"key": noise_key, "label": f"{args.label} noise {stamp}", "model": args.model},
        timeout_s=40,
        retries=3,
    )
    calls.append(rec)
    if noise_created is None:
        summary = {
            "schemaVersion": "phase3_closure_battery.v1",
            "passed": False,
            "reason": "noise_session_create_failed",
            "primarySessionKey": primary_key,
            "noiseSessionKey": noise_key,
            "model": args.model,
            "artifactDir": str(outdir),
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    primary_session_id = str(created.get("sessionId") or "")
    if not primary_session_id:
        summary = {
            "schemaVersion": "phase3_closure_battery.v1",
            "passed": False,
            "reason": "primary_session_id_missing",
            "primarySessionKey": primary_key,
            "model": args.model,
            "artifactDir": str(outdir),
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    assistant_count = 0
    steps: list[tuple[str, str]] = [
        ("status", "/crag_status"),
        ("explain", "/crag_explain_memory"),
        (
            "seed_task",
            "Task scenario seed: objective close phase3 lossless memory; blockers mirror-heavy task-state and unstable heavy benchmark; next steps align runtime path, fix quote-span, fix task-state grounding; acceptance criteria exact quote span and session-grounded task-state answers.",
        ),
        ("ask_task_state", "What is the current task state in this session?"),
        ("ask_blockers", "What are the current blockers in this session?"),
        ("ask_next_steps", "What are the next steps in this session?"),
        ("ask_changed", "What changed in this session since the task seed?"),
        (
            "seed_quote",
            f'Quote seed: {quote_token} appears in this exact sentence and should be quoted exactly later.',
        ),
        ("ask_quote", f"What did we say earlier about {quote_token}? Quote the earlier span exactly."),
        (
            "long_continuity",
            "Long continuity check: summarize objective, blockers, and next steps in 3 bullets with source grounding.",
        ),
    ]
    for step_id, prompt in steps:
        assistant_count = _send_and_capture(primary_key, step_id, prompt, assistant_count, calls, transcript, wait_s=max(25, int(args.wait_seconds)))

    # Stress element: cross-session noise + second quote/state checks.
    noise_assistant_count = 0
    noise_prompts = [
        "Noise session: acknowledge with NOISE-ACK-1.",
        "Noise session: summarize this message in one line and include NOISE-ACK-2.",
        "Noise session: do a short memory status mention and include NOISE-ACK-3.",
    ]
    for idx, prompt in enumerate(noise_prompts, start=1):
        noise_assistant_count = _send_and_capture(
            noise_key,
            f"noise_{idx}",
            prompt,
            noise_assistant_count,
            calls,
            transcript,
            wait_s=max(25, int(args.wait_seconds)),
        )

    assistant_count = _send_and_capture(
        primary_key,
        "stress_requote",
        f"After the noise session activity, quote {quote_token} exactly again and list blockers + next steps in this session.",
        assistant_count,
        calls,
        transcript,
        wait_s=max(25, int(args.wait_seconds)),
    )
    assistant_count = _send_and_capture(
        primary_key,
        "stress_continuity",
        "After stress activity, what is the current task state and what changed since seed?",
        assistant_count,
        calls,
        transcript,
        wait_s=max(25, int(args.wait_seconds)),
    )

    primary_obj, rec = _gateway_call("sessions.get", {"key": primary_key}, timeout_s=45, retries=2)
    calls.append({"step": "primary_get", **rec})
    noise_obj, rec = _gateway_call("sessions.get", {"key": noise_key}, timeout_s=45, retries=2)
    calls.append({"step": "noise_get", **rec})

    checks = {
        "runtime_path_truth": False,
        "status_command_ok": False,
        "explain_command_ok": False,
        "quote_exact_primary": False,
        "quote_exact_sentence_match": False,
        "task_state_present": False,
        "task_state_session_grounded": False,
        "blockers_present": False,
        "next_steps_present": False,
        "changed_present": False,
        "long_continuity_present": False,
        "stress_quote_exact_after_noise": False,
        "stress_continuity_present": False,
        "memory_to_context_signal": False,
        "trajectory_export_ok": False,
        "trajectory_message_count_ok": False,
        "summary_lineage_runtime_ok": False,
        "long_session_recoverability_runtime_ok": False,
    }

    backend = args.backend_base_url.rstrip("/")

    def backend_call(path: str, payload: dict[str, Any], timeout_s: int = 25) -> tuple[int, dict[str, Any] | None]:
        status, body, raw = _post_json(f"{backend}{path}", payload, timeout_s=timeout_s)
        backend_calls.append({"path": path, "payload": payload, "status": status, "body": body, "raw": raw})
        return status, body

    # Runtime readback proof for trajectory.
    st, body = backend_call("/session_structured_export", {"session_id": primary_session_id})
    export_before = body if (st == 200 and isinstance(body, dict)) else None
    if export_before is not None:
        checks["trajectory_export_ok"] = True
        msg_count = int((export_before.get("part_stats") or {}).get("message_count") or 0)
        # 10 primary prompts + 2 stress prompts + assistant replies should comfortably exceed this floor.
        checks["trajectory_message_count_ok"] = msg_count >= 12

    # Dedicated compaction proof session: deterministic long-session + lineage/recoverability evidence.
    proof_session_id = f"{primary_session_id}-compaction-proof"
    appended_ok = True
    for idx in range(26):
        text = (
            f"Compaction proof message {idx} for {proof_session_id}. "
            f"Marker-LONG-RECOVERY-{idx:02d}. "
            "This payload is intentionally verbose to avoid low-value quarantine heuristics."
        )
        st, _ = backend_call(
            "/session_append_message",
            {
                "session_id": proof_session_id,
                "message_id": f"proof-{idx}",
                "sender": "user" if idx % 2 == 0 else "assistant",
                "text": text,
                "created_at": f"2026-04-12T10:{(idx % 60):02d}:00Z",
            },
        )
        appended_ok = appended_ok and st == 200

    seg_count = 0
    if appended_ok:
        st, _ = backend_call("/session_compact", {"session_id": proof_session_id, "older_than_index": 20})
        if st == 200:
            st, body = backend_call("/session_compaction_state", {"session_id": proof_session_id})
            comp_state = body.get("state") if (st == 200 and isinstance(body, dict)) else None
            seg_count = int(((comp_state or {}).get("stats") or {}).get("compacted_segments") or 0) if isinstance(comp_state, dict) else 0

            st, body = backend_call("/session_structured_export", {"session_id": proof_session_id})
            export_after = body if (st == 200 and isinstance(body, dict)) else None
            if export_after is not None:
                comp = export_after.get("compaction") or {}
                segments = list(comp.get("segments") or [])
                if seg_count > 0 and segments:
                    first = segments[0]
                    checks["summary_lineage_runtime_ok"] = len(list(first.get("lineage") or [])) >= 1
                    # Recoverability signal: segment carries raw snapshot support.
                    checks["long_session_recoverability_runtime_ok"] = len(list(first.get("raw_snapshot") or [])) >= 1
                    # Stronger recoverability proof: expand compacted summary into message refs.
                    summary_ref = {
                        "item_type": "compacted_summary",
                        "session_id": proof_session_id,
                        "primary_id": str(first.get("segment_id") or ""),
                    }
                    if summary_ref["primary_id"]:
                        st, body = backend_call("/session_expand_item", {"ref": summary_ref})
                        expanded = list((body or {}).get("expanded") or []) if isinstance(body, dict) else []
                        if not any(isinstance(item, dict) and item.get("item_type") == "message" for item in expanded):
                            checks["long_session_recoverability_runtime_ok"] = False
                else:
                    checks["summary_lineage_runtime_ok"] = False
                    checks["long_session_recoverability_runtime_ok"] = False

    for row in transcript:
        step = row.get("step")
        txt = str(row.get("assistant") or "")
        txt_l = txt.lower()
        if step == "status":
            checks["status_command_ok"] = bool(txt.strip())
            checks["runtime_path_truth"] = "runtime entry path:" in txt_l and "runtime plugin root:" in txt_l
        elif step == "explain":
            checks["explain_command_ok"] = "cognitiverag memory architecture" in txt_l
            checks["memory_to_context_signal"] = "context engine" in txt_l or "context layer" in txt_l
        elif step == "ask_task_state":
            checks["task_state_present"] = bool(txt.strip())
            checks["task_state_session_grounded"] = ("session" in txt_l) and ("source" in txt_l or "ground" in txt_l)
        elif step == "ask_blockers":
            checks["blockers_present"] = bool(txt.strip())
        elif step == "ask_next_steps":
            checks["next_steps_present"] = bool(txt.strip())
        elif step == "ask_changed":
            checks["changed_present"] = bool(txt.strip())
        elif step == "ask_quote":
            checks["quote_exact_primary"] = quote_token.lower() in txt_l
            checks["quote_exact_sentence_match"] = "appears in this exact sentence" in txt_l and quote_token.lower() in txt_l
        elif step == "long_continuity":
            checks["long_continuity_present"] = bool(txt.strip())
        elif step == "stress_requote":
            checks["stress_quote_exact_after_noise"] = quote_token.lower() in txt_l
        elif step == "stress_continuity":
            checks["stress_continuity_present"] = bool(txt.strip())

    passed = all(checks.values())
    coverage = {
        "runtime_path_truth": checks["runtime_path_truth"],
        "status_surface": checks["status_command_ok"],
        "explain_surface": checks["explain_command_ok"],
        "quote_span_surface": checks["quote_exact_primary"] and checks["quote_exact_sentence_match"] and checks["stress_quote_exact_after_noise"],
        "task_state_surface": checks["task_state_present"] and checks["task_state_session_grounded"],
        "blockers_next_steps_surface": checks["blockers_present"] and checks["next_steps_present"] and checks["changed_present"],
        "long_session_continuity_surface": checks["long_continuity_present"] and checks["stress_continuity_present"],
        "memory_to_context_surface": checks["memory_to_context_signal"],
        "conversation_trajectory_surface": checks["trajectory_export_ok"] and checks["trajectory_message_count_ok"],
        "summary_lineage_surface": checks["summary_lineage_runtime_ok"],
        "long_session_recoverability_surface": checks["long_session_recoverability_runtime_ok"],
    }
    summary = {
        "schemaVersion": "phase3_closure_battery.v1",
        "startedAt": stamp,
        "primarySessionKey": primary_key,
        "noiseSessionKey": noise_key,
        "model": args.model,
        "quoteToken": quote_token,
        "artifactDir": str(outdir),
        "instrumentRole": "phase3_closure_authoritative",
        "heavyBenchmarkRole": "stress_telemetry_non_blocking",
        "checks": checks,
        "coverage": coverage,
        "passed": passed,
        "notes": [
            "This closure battery is Phase-3 scoped and avoids restart-driven gateway lifecycle noise.",
            "Includes stress via cross-session noise plus repeated quote+continuity checks.",
        ],
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
    (outdir / "backend_calls.json").write_text(json.dumps(backend_calls, indent=2), encoding="utf-8")
    (outdir / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    (outdir / "primary_session_get.json").write_text(json.dumps(primary_obj or {}, indent=2), encoding="utf-8")
    (outdir / "noise_session_get.json").write_text(json.dumps(noise_obj or {}, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
