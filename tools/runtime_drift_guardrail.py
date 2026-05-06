#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(["git", *args], cwd=str(cwd), text=True, capture_output=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def collect_guardrail_report(
    *,
    plugin_root: Path,
    backend_root: Path,
    runtime_root: Path,
    plugin_default_branch: str = "main",
    backend_default_branch: str = "master",
    mode: str = "audit",
    allow_report_only_success: bool = False,
    expected_plugin_tag: str | None = None,
    expected_backend_tag: str | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []

    plugin_rc, plugin_head, _ = _run_git(["rev-parse", "HEAD"], plugin_root)
    plugin_branch_rc, plugin_branch, _ = _run_git(["branch", "--show-current"], plugin_root)
    plugin_origin_rc, plugin_origin_head, _ = _run_git(["rev-parse", f"origin/{plugin_default_branch}"], plugin_root)
    plugin_status_rc, plugin_status_out, _ = _run_git(["status", "--porcelain"], plugin_root)
    plugin_tag_sha = None
    if expected_plugin_tag:
        tag_rc, tag_out, _ = _run_git(["rev-list", "-n", "1", expected_plugin_tag], plugin_root)
        if tag_rc == 0 and tag_out:
            plugin_tag_sha = tag_out

    backend_rc, backend_head, _ = _run_git(["rev-parse", "HEAD"], backend_root)
    backend_branch_rc, backend_branch, _ = _run_git(["branch", "--show-current"], backend_root)
    backend_origin_rc, backend_origin_head, _ = _run_git(["rev-parse", f"origin/{backend_default_branch}"], backend_root)
    backend_tag_sha = None
    if expected_backend_tag:
        tag_rc, tag_out, _ = _run_git(["rev-list", "-n", "1", expected_backend_tag], backend_root)
        if tag_rc == 0 and tag_out:
            backend_tag_sha = tag_out

    repo_index = plugin_root / "index.ts"
    runtime_index = runtime_root / "index.ts"
    repo_hash = _sha256(repo_index)
    runtime_hash = _sha256(runtime_index)
    parity_match = repo_hash is not None and runtime_hash is not None and repo_hash == runtime_hash

    plugin_dirty = bool(plugin_status_out.strip()) if plugin_status_rc == 0 else True
    plugin_head_matches_origin = plugin_rc == 0 and plugin_origin_rc == 0 and plugin_head == plugin_origin_head
    backend_head_matches_origin = backend_rc == 0 and backend_origin_rc == 0 and backend_head == backend_origin_head
    plugin_on_default_branch = plugin_branch_rc == 0 and plugin_branch == plugin_default_branch
    backend_on_default_branch = backend_branch_rc == 0 and backend_branch == backend_default_branch

    if mode == "release-signoff":
        if plugin_dirty:
            failures.append({"code": "PLUGIN_DIRTY_WORKTREE", "reason": "plugin worktree is dirty in release/signoff mode"})
        if not plugin_on_default_branch:
            failures.append({"code": "PLUGIN_WRONG_BRANCH", "reason": f"plugin not on default branch {plugin_default_branch}"})
        if not backend_on_default_branch:
            failures.append({"code": "BACKEND_WRONG_BRANCH", "reason": f"backend not on default branch {backend_default_branch}"})
        if not plugin_head_matches_origin:
            failures.append({"code": "PLUGIN_HEAD_ORIGIN_MISMATCH", "reason": "plugin HEAD does not match origin default branch"})
        if not backend_head_matches_origin:
            failures.append({"code": "BACKEND_HEAD_ORIGIN_MISMATCH", "reason": "backend HEAD does not match origin default branch"})
        if not parity_match:
            failures.append({"code": "RUNTIME_PLUGIN_PARITY_MISMATCH", "reason": "runtime extension index.ts differs from plugin repo index.ts"})
        if expected_plugin_tag and plugin_tag_sha and plugin_head != plugin_tag_sha:
            failures.append({"code": "PLUGIN_TAG_MISMATCH", "reason": f"plugin HEAD does not match expected tag {expected_plugin_tag}"})
        if expected_backend_tag and backend_tag_sha and backend_head != backend_tag_sha:
            failures.append({"code": "BACKEND_TAG_MISMATCH", "reason": f"backend HEAD does not match expected tag {expected_backend_tag}"})

    report = {
        "schemaVersion": "runtime-drift-guardrail.v1",
        "mode": mode,
        "allowReportOnlySuccess": allow_report_only_success,
        "plugin": {
            "root": str(plugin_root),
            "defaultBranch": plugin_default_branch,
            "currentBranch": plugin_branch if plugin_branch_rc == 0 else None,
            "head": plugin_head if plugin_rc == 0 else None,
            "originDefaultHead": plugin_origin_head if plugin_origin_rc == 0 else None,
            "headMatchesOriginDefault": plugin_head_matches_origin,
            "dirty": plugin_dirty,
            "dirtyEntries": plugin_status_out.splitlines() if plugin_status_out else [],
            "expectedTag": expected_plugin_tag,
            "expectedTagSha": plugin_tag_sha,
        },
        "backend": {
            "root": str(backend_root),
            "defaultBranch": backend_default_branch,
            "currentBranch": backend_branch if backend_branch_rc == 0 else None,
            "head": backend_head if backend_rc == 0 else None,
            "originDefaultHead": backend_origin_head if backend_origin_rc == 0 else None,
            "headMatchesOriginDefault": backend_head_matches_origin,
            "expectedTag": expected_backend_tag,
            "expectedTagSha": backend_tag_sha,
        },
        "runtimeParity": {
            "runtimeRoot": str(runtime_root),
            "repoIndexPath": str(repo_index),
            "runtimeIndexPath": str(runtime_index),
            "repoIndexSha256": repo_hash,
            "runtimeIndexSha256": runtime_hash,
            "matches": parity_match,
        },
        "reasonCodes": [x["code"] for x in failures],
        "failures": failures,
    }
    if mode == "audit":
        report["passed"] = True if allow_report_only_success else not failures
    else:
        report["passed"] = not failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime/repo drift guardrail for release/signoff and audit/report modes.")
    parser.add_argument("--plugin-root", default="/home/ictin_claw/.openclaw/workspace/openclaw-cognitiverag-memory")
    parser.add_argument("--backend-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--runtime-root", default="/home/ictin_claw/.openclaw/workspace/.openclaw/extensions/cognitiverag-memory")
    parser.add_argument("--plugin-default-branch", default="main")
    parser.add_argument("--backend-default-branch", default="master")
    parser.add_argument("--mode", choices=["release-signoff", "audit"], default="audit")
    parser.add_argument("--allow-report-only-success", action="store_true")
    parser.add_argument("--expected-plugin-tag", default=None)
    parser.add_argument("--expected-backend-tag", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    report = collect_guardrail_report(
        plugin_root=Path(args.plugin_root).resolve(),
        backend_root=Path(args.backend_root).resolve(),
        runtime_root=Path(args.runtime_root).resolve(),
        plugin_default_branch=args.plugin_default_branch,
        backend_default_branch=args.backend_default_branch,
        mode=args.mode,
        allow_report_only_success=args.allow_report_only_success,
        expected_plugin_tag=args.expected_plugin_tag,
        expected_backend_tag=args.expected_backend_tag,
    )
    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
