#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from accepted_feature_gate_lib import load_manifest, utc_stamp, validate_manifest_shape


def main() -> int:
    parser = argparse.ArgumentParser(description="Run combined accepted-feature dual-surface gate.")
    parser.add_argument("--manifest", default="tools/accepted_feature_regression_manifest.json")
    parser.add_argument("--plugin-root", default="/home/ictin_claw/.openclaw/workspace/openclaw-cognitiverag-memory")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    stamp = utc_stamp()
    outdir = Path(args.output_dir) if args.output_dir else (repo_root / "forensics" / f"{stamp}_accepted_feature_dual_surface_gate")
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    shape_failures, by_id = validate_manifest_shape(manifest)

    backend_outdir = outdir / "backend"
    plugin_outdir = outdir / "plugin"
    backend_outdir.mkdir(parents=True, exist_ok=True)
    plugin_outdir.mkdir(parents=True, exist_ok=True)

    backend_cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        str(repo_root / "tools" / "run_accepted_feature_backend_gate.py"),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(backend_outdir),
    ]
    plugin_cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        str(repo_root / "tools" / "run_accepted_feature_plugin_gate.py"),
        "--manifest",
        str(manifest_path),
        "--plugin-root",
        str(args.plugin_root),
        "--output-dir",
        str(plugin_outdir),
    ]

    import subprocess

    backend_proc = subprocess.run(backend_cmd, cwd=str(repo_root), text=True, capture_output=True)
    plugin_proc = subprocess.run(plugin_cmd, cwd=str(repo_root), text=True, capture_output=True)

    backend_report_path = backend_outdir / "backend_gate_report.json"
    plugin_report_path = plugin_outdir / "plugin_gate_report.json"
    backend_report = json.loads(backend_report_path.read_text(encoding="utf-8")) if backend_report_path.exists() else None
    plugin_report = json.loads(plugin_report_path.read_text(encoding="utf-8")) if plugin_report_path.exists() else None

    missing_coverage = []
    if by_id:
        for fid, row in by_id.items():
            if not row.get("backendTests"):
                missing_coverage.append(f"{fid}: missing backendTests")
            if not row.get("pluginTests") and not str(row.get("pluginNoSurfaceJustification") or "").strip():
                missing_coverage.append(f"{fid}: missing pluginTests and pluginNoSurfaceJustification")

    failures = []
    failures.extend(shape_failures)
    failures.extend(missing_coverage)
    if backend_proc.returncode != 0:
        failures.append("backend gate failed")
    if plugin_proc.returncode != 0:
        failures.append("plugin gate failed")

    combined = {
        "schemaVersion": "accepted-feature-dual-surface-gate.v1",
        "manifestPath": str(manifest_path),
        "artifactDir": str(outdir),
        "standaloneBackend": {
            "returncode": backend_proc.returncode,
            "reportPath": str(backend_report_path),
            "passed": bool(backend_report and backend_report.get("passed")),
            "stdout_tail": "\n".join(backend_proc.stdout.splitlines()[-40:]),
            "stderr_tail": "\n".join(backend_proc.stderr.splitlines()[-40:]),
        },
        "pluginIntegration": {
            "returncode": plugin_proc.returncode,
            "reportPath": str(plugin_report_path),
            "passed": bool(plugin_report and plugin_report.get("passed")),
            "stdout_tail": "\n".join(plugin_proc.stdout.splitlines()[-40:]),
            "stderr_tail": "\n".join(plugin_proc.stderr.splitlines()[-40:]),
        },
        "coverage": {
            "featureCount": len(by_id),
            "missingCoverage": missing_coverage,
        },
        "failures": failures,
        "passed": not failures,
    }
    out_json = outdir / "combined_gate_report.json"
    out_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    if missing_coverage:
        (outdir / "missing_coverage_report.json").write_text(
            json.dumps({"missingCoverage": missing_coverage, "passed": False}, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({"passed": combined["passed"], "report": str(out_json), "failures": failures}, indent=2))
    return 0 if combined["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
