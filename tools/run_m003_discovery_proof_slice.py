#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CognitiveRAG.session_memory import context_window


EXACT_PROMPT = "What else matters here that I may not be asking?"


def _seed_fixture(session_id: str) -> None:
    os.makedirs(context_window.WORKDIR, exist_ok=True)
    rows = [
        {"index": 0, "text": "Feature flag is enabled in production for all users."},
        {"index": 1, "text": "Feature flag is not enabled in production for all users."},
        {"index": 2, "text": "Rollback rehearsal was skipped in the last rollout drill."},
        {"index": 3, "text": "Missing rollback rehearsal increases latent incident risk."},
        {"index": 4, "text": "Deployment looked successful in initial smoke checks."},
        {"index": 5, "text": "Audit note: safety gate depends on contradiction visibility."},
        {"index": 6, "text": "Non-obvious risk: stale fallback assumptions can hide unresolved blockers."},
        {"index": 7, "text": "Operator requested bounded discovery with explicit provenance."},
        {"index": 8, "text": "A previous run skipped contradiction checks due haste."},
        {"index": 9, "text": "A current run requires contradiction checks before rollout approval."},
        {"index": 10, "text": "Unknown unknown probe: verify hidden dependencies before cutover."},
        {"index": 11, "text": "Context must stay bounded and source-truthful even under pressure."},
    ]
    with open(os.path.join(context_window.WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def _synthesize_answer(out: dict) -> str:
    discovery = out.get("discovery") or {}
    injected = list(discovery.get("injected_discoveries") or [])
    contradictions = list(discovery.get("contradictions") or [])
    findings: list[str] = []
    seen = set()
    for item in injected:
        text = str(item.get("text", "")).strip()
        key = " ".join(text.lower().split())
        if not text or key in seen:
            continue
        seen.add(key)
        findings.append(text)
        if len(findings) >= 2:
            break
    contradiction_note = "No contradiction surfaced."
    if contradictions:
        c0 = contradictions[0]
        contradiction_note = (
            f"Contradiction detected: {c0.get('left_evidence_id')} vs {c0.get('right_evidence_id')} "
            f"({c0.get('reason')})."
        )
    weak_signal_count = len(list((discovery.get("ledger") or {}).get("rejected_branches", []))) + len(
        list((discovery.get("ledger") or {}).get("unresolved_questions", []))
    )
    if not findings:
        findings = ["No non-obvious finding passed bounded discovery thresholds."]
    lines = ["Bounded discovery findings:"]
    lines.extend(f"- {text}" for text in findings)
    lines.append(f"- {contradiction_note}")
    if weak_signal_count > 0:
        lines.append(f"- Weak-signal artifact: {weak_signal_count} bounded branch/review warning(s) surfaced.")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser("run_m003_discovery_proof_slice")
    parser.add_argument("--tag", default="m003_discovery_proof_slice")
    parser.add_argument("--session-id", default="m003-discovery-proof")
    parser.add_argument("--budget", type=int, default=1400)
    parser.add_argument("--fresh-tail-count", type=int, default=12)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "forensics" / f"{ts}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    context_window.WORKDIR = str(out_dir / "workdir")
    os.makedirs(context_window.WORKDIR, exist_ok=True)

    _seed_fixture(args.session_id)
    out = context_window.assemble_context(
        session_id=args.session_id,
        fresh_tail_count=int(args.fresh_tail_count),
        budget=int(args.budget),
        query=EXACT_PROMPT,
    )

    discovery = out.get("discovery", {})
    explanation = out.get("explanation", {})
    selected_discovery_blocks = [b for b in explanation.get("selected_blocks", []) if b.get("lane") == "discovery"]
    contradiction_payload = {
        "contradictions": discovery.get("contradictions", []),
        "ledger": discovery.get("ledger", {}),
        "contradiction_count": len(discovery.get("contradictions", [])),
        "weak_signal_artifacts": {
            "rejected_branches": list((discovery.get("ledger") or {}).get("rejected_branches", [])),
            "unresolved_questions": list((discovery.get("ledger") or {}).get("unresolved_questions", [])),
            "weak_signal_count": len(list((discovery.get("ledger") or {}).get("rejected_branches", [])))
            + len(list((discovery.get("ledger") or {}).get("unresolved_questions", []))),
        },
    }
    prompt_answer_payload = {
        "prompt": EXACT_PROMPT,
        "answer": _synthesize_answer(out),
        "discovery_ids": [item.get("discovery_id") for item in discovery.get("injected_discoveries", [])],
    }
    discovery_trace_payload = {
        "prompt": EXACT_PROMPT,
        "discovery_plan": out.get("discovery_plan", {}),
        "discovery": discovery,
        "selector_metrics_discovery": (out.get("selector_metrics") or {}).get("discovery", {}),
        "selected_discovery_blocks": selected_discovery_blocks,
        "bounded_check": {
            "bounded": bool(discovery.get("bounded", False)),
            "used_tokens": int(discovery.get("used_tokens", 0)),
            "budget_tokens": int(discovery.get("budget_tokens", 0)),
            "within_budget": int(discovery.get("used_tokens", 0)) <= int(discovery.get("budget_tokens", 0)),
        },
    }
    explanation_payload = {
        "prompt": EXACT_PROMPT,
        "explanation": explanation,
        "discovery_contribution_truth": {
            "selected_discovery_block_count": len(selected_discovery_blocks),
            "discovery_injected_count": len(discovery.get("injected_discoveries", [])),
            "contribution_visible": bool(selected_discovery_blocks),
        },
    }

    _write_json(out_dir / "discovery_trace_artifact.json", discovery_trace_payload)
    _write_json(out_dir / "contradiction_artifact.json", contradiction_payload)
    _write_json(out_dir / "prompt_answer_capture.json", prompt_answer_payload)
    _write_json(out_dir / "explanation_discovery_artifact.json", explanation_payload)

    summary = {
        "artifact_dir": str(out_dir),
        "prompt": EXACT_PROMPT,
        "bounded": discovery_trace_payload["bounded_check"]["bounded"],
        "within_budget": discovery_trace_payload["bounded_check"]["within_budget"],
        "contradiction_count": contradiction_payload["contradiction_count"],
        "weak_signal_count": contradiction_payload["weak_signal_artifacts"]["weak_signal_count"],
        "selected_discovery_block_count": len(selected_discovery_blocks),
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
