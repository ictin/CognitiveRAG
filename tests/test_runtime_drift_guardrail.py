from __future__ import annotations

import subprocess
from pathlib import Path

from tools.runtime_drift_guardrail import collect_guardrail_report


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, text=True, capture_output=True)


def _make_repo(path: Path, *, branch: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-b", branch)
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "tests")
    (path / "index.ts").write_text("export const v = 1;\n", encoding="utf-8")
    _git(path, "add", "index.ts")
    _git(path, "commit", "-m", "init")
    _git(path, "remote", "add", "origin", str(path))
    _git(path, "fetch", "origin")


def test_release_signoff_clean_state_passes(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    backend = tmp_path / "backend"
    runtime = tmp_path / "runtime"
    _make_repo(plugin, branch="main")
    _make_repo(backend, branch="master")
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "index.ts").write_text((plugin / "index.ts").read_text(encoding="utf-8"), encoding="utf-8")

    report = collect_guardrail_report(
        plugin_root=plugin,
        backend_root=backend,
        runtime_root=runtime,
        mode="release-signoff",
    )
    assert report["passed"] is True
    assert report["reasonCodes"] == []


def test_release_signoff_dirty_workspace_fails(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    backend = tmp_path / "backend"
    runtime = tmp_path / "runtime"
    _make_repo(plugin, branch="main")
    _make_repo(backend, branch="master")
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "index.ts").write_text((plugin / "index.ts").read_text(encoding="utf-8"), encoding="utf-8")
    (plugin / "dirty.txt").write_text("x\n", encoding="utf-8")

    report = collect_guardrail_report(
        plugin_root=plugin,
        backend_root=backend,
        runtime_root=runtime,
        mode="release-signoff",
    )
    assert report["passed"] is False
    assert "PLUGIN_DIRTY_WORKTREE" in report["reasonCodes"]


def test_release_signoff_parity_mismatch_fails(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    backend = tmp_path / "backend"
    runtime = tmp_path / "runtime"
    _make_repo(plugin, branch="main")
    _make_repo(backend, branch="master")
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "index.ts").write_text("export const v = 999;\n", encoding="utf-8")

    report = collect_guardrail_report(
        plugin_root=plugin,
        backend_root=backend,
        runtime_root=runtime,
        mode="release-signoff",
    )
    assert report["passed"] is False
    assert "RUNTIME_PLUGIN_PARITY_MISMATCH" in report["reasonCodes"]


def test_audit_report_mode_can_exit_success_when_explicit(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    backend = tmp_path / "backend"
    runtime = tmp_path / "runtime"
    _make_repo(plugin, branch="main")
    _make_repo(backend, branch="master")
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "index.ts").write_text("export const v = 999;\n", encoding="utf-8")
    (plugin / "dirty.txt").write_text("x\n", encoding="utf-8")

    report = collect_guardrail_report(
        plugin_root=plugin,
        backend_root=backend,
        runtime_root=runtime,
        mode="audit",
        allow_report_only_success=True,
    )
    assert report["passed"] is True
    assert report["plugin"]["dirty"] is True
    assert report["runtimeParity"]["matches"] is False
