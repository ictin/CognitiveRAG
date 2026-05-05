#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Run accepted-feature backend regression gate (F-001..F-025).")
    parser.add_argument("--manifest", default="tools/accepted_feature_regression_manifest.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    stamp = utc_stamp()
    outdir = Path(args.output_dir) if args.output_dir else (repo_root / "forensics" / f"{stamp}_accepted_feature_backend_gate")
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    failures, by_id = validate_manifest_shape(manifest)

    backend_tests: list[str] = []
    for fid in FEATURE_IDS:
        backend_tests.extend(by_id.get(fid, {}).get("backendTests", []))
    backend_tests = sorted(dict.fromkeys(backend_tests))

    missing_files = check_paths_exist(repo_root, backend_tests)
    if missing_files:
        failures.append(f"missing backend test files: {missing_files}")

    cmd_result = None
    if not failures:
        cmd = [str(repo_root / ".venv" / "bin" / "python"), "-m", "pytest", *backend_tests]
        cmd_result = run_cmd(cmd, cwd=repo_root)
        if not cmd_result.ok:
            failures.append("pytest backend gate failed")

    feature_rows = []
    for fid in FEATURE_IDS:
        row = by_id[fid]
        feature_rows.append(
            {
                "id": fid,
                "surface": row["surface"],
                "backendTests": row["backendTests"],
                "backendCoverageOk": bool(row["backendTests"]),
                "proofType": row["proofType"],
            }
        )

    report = {
        "schemaVersion": "accepted-feature-backend-gate.v1",
        "manifestPath": str(manifest_path),
        "artifactDir": str(outdir),
        "featureCount": len(feature_rows),
        "features": feature_rows,
        "backendTestsCount": len(backend_tests),
        "failures": failures,
        "backendCommand": summarize_result(cmd_result) if cmd_result else None,
        "passed": not failures,
    }

    out_json = outdir / "backend_gate_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(out_json), "failures": failures}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
