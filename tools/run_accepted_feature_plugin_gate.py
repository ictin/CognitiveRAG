#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from accepted_feature_gate_lib import (
    FEATURE_IDS,
    check_paths_exist,
    load_manifest,
    run_cmd,
    summarize_result,
    utc_stamp,
    validate_manifest_shape,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run accepted-feature plugin/runtime regression gate (F-001..F-025).")
    parser.add_argument("--manifest", default="tools/accepted_feature_regression_manifest.json")
    parser.add_argument(
        "--plugin-root",
        default="/home/ictin_claw/.openclaw/workspace/openclaw-cognitiverag-memory",
    )
    parser.add_argument(
        "--runtime-root",
        default="/home/ictin_claw/.openclaw/workspace/.openclaw/extensions/cognitiverag-memory",
    )
    parser.add_argument("--drift-guardrail-mode", choices=["off", "audit", "release-signoff"], default="off")
    parser.add_argument("--drift-guardrail-allow-report-only-success", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    plugin_root = Path(args.plugin_root).resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    stamp = utc_stamp()
    outdir = Path(args.output_dir) if args.output_dir else (repo_root / "forensics" / f"{stamp}_accepted_feature_plugin_gate")
    outdir.mkdir(parents=True, exist_ok=True)
    runtime_ext_root = Path(args.runtime_root).resolve()

    manifest = load_manifest(manifest_path)
    failures, by_id = validate_manifest_shape(manifest)

    plugin_tests: list[str] = []
    for fid in FEATURE_IDS:
        plugin_tests.extend(by_id.get(fid, {}).get("pluginTests", []))
    plugin_tests = sorted(dict.fromkeys(plugin_tests))

    missing_files = check_paths_exist(plugin_root, plugin_tests)
    if missing_files:
        failures.append(f"missing plugin test files: {missing_files}")

    results = []
    if not failures:
        for rel in plugin_tests:
            cmd = ["npx", "tsx", rel]
            result = run_cmd(cmd, cwd=plugin_root)
            results.append({"test": rel, **summarize_result(result)})
            if not result.ok:
                failures.append(f"plugin test failed: {rel}")

    # runtime parity check where available (required by this gate)
    runtime_parity = {
        "runtimeExtensionRoot": str(runtime_ext_root),
        "runtimeExtensionExists": runtime_ext_root.exists(),
        "repoIndexExists": (plugin_root / "index.ts").exists(),
        "runtimeIndexExists": (runtime_ext_root / "index.ts").exists(),
        "matchesRepo": None,
    }
    if runtime_parity["repoIndexExists"] and runtime_parity["runtimeIndexExists"]:
        runtime_parity["matchesRepo"] = (plugin_root / "index.ts").read_bytes() == (runtime_ext_root / "index.ts").read_bytes()
    if runtime_parity["runtimeExtensionExists"] and runtime_parity["matchesRepo"] is False:
        failures.append("runtime/plugin parity mismatch: runtime extension index.ts differs from plugin repo index.ts")

    drift_guardrail = None
    if args.drift_guardrail_mode != "off":
        drift_json = outdir / "drift_guardrail_report.json"
        guardrail_cmd = [
            str(repo_root / ".venv" / "bin" / "python"),
            str(repo_root / "tools" / "runtime_drift_guardrail.py"),
            "--plugin-root",
            str(plugin_root),
            "--backend-root",
            str(repo_root),
            "--runtime-root",
            str(runtime_ext_root),
            "--mode",
            args.drift_guardrail_mode,
            "--json-out",
            str(drift_json),
        ]
        if args.drift_guardrail_allow_report_only_success:
            guardrail_cmd.append("--allow-report-only-success")
        proc = subprocess.run(guardrail_cmd, cwd=str(repo_root), text=True, capture_output=True)
        if drift_json.exists():
            drift_guardrail = json.loads(drift_json.read_text(encoding="utf-8"))
        else:
            drift_guardrail = {
                "passed": False,
                "reasonCodes": ["DRIFT_GUARDRAIL_REPORT_MISSING"],
                "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
                "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
            }
        if proc.returncode != 0:
            failures.append(f"drift guardrail failed in mode={args.drift_guardrail_mode}")

    feature_rows = []
    for fid in FEATURE_IDS:
        row = by_id[fid]
        feature_rows.append(
            {
                "id": fid,
                "surface": row["surface"],
                "pluginTests": row["pluginTests"],
                "pluginNoSurfaceJustification": row.get("pluginNoSurfaceJustification"),
                "pluginCoverageOk": bool(row["pluginTests"]) or bool(str(row.get("pluginNoSurfaceJustification") or "").strip()),
                "proofType": row["proofType"],
            }
        )

    report = {
        "schemaVersion": "accepted-feature-plugin-gate.v1",
        "manifestPath": str(manifest_path),
        "pluginRoot": str(plugin_root),
        "artifactDir": str(outdir),
        "featureCount": len(feature_rows),
        "features": feature_rows,
        "pluginTestsCount": len(plugin_tests),
        "runtimeParity": runtime_parity,
        "driftGuardrail": drift_guardrail,
        "testRuns": results,
        "failures": failures,
        "passed": not failures,
    }

    out_json = outdir / "plugin_gate_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(out_json), "failures": failures}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
