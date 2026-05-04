from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FEATURE_IDS = [f"F-{i:03d}" for i in range(1, 21)]


@dataclass
class CommandResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest_shape(manifest: Dict) -> Tuple[List[str], Dict[str, Dict]]:
    failures: List[str] = []
    entries = manifest.get("acceptedFeatures")
    if not isinstance(entries, list):
        return (["manifest.acceptedFeatures must be a list"], {})

    by_id: Dict[str, Dict] = {}
    for row in entries:
        fid = str(row.get("id") or "").strip()
        if not fid:
            failures.append("feature row missing id")
            continue
        if fid in by_id:
            failures.append(f"duplicate feature id {fid}")
            continue
        by_id[fid] = row

    missing = [fid for fid in FEATURE_IDS if fid not in by_id]
    extra = [fid for fid in by_id.keys() if fid not in FEATURE_IDS]
    if missing:
        failures.append(f"missing feature IDs: {missing}")
    if extra:
        failures.append(f"unexpected feature IDs: {extra}")

    for fid in FEATURE_IDS:
        row = by_id.get(fid)
        if not row:
            continue
        surface = str(row.get("surface") or "")
        if surface not in {"backend-only", "plugin-only", "dual"}:
            failures.append(f"{fid}: invalid surface {surface!r}")
        backend_tests = row.get("backendTests")
        plugin_tests = row.get("pluginTests")
        if not isinstance(backend_tests, list) or not all(isinstance(x, str) and x.strip() for x in backend_tests):
            failures.append(f"{fid}: backendTests must be non-empty string list")
        if not isinstance(plugin_tests, list) or not all(isinstance(x, str) and x.strip() for x in plugin_tests):
            failures.append(f"{fid}: pluginTests must be non-empty string list")
        if not backend_tests:
            failures.append(f"{fid}: backendTests empty (fail closed)")
        if not plugin_tests:
            failures.append(f"{fid}: pluginTests empty (fail closed)")
        if not str(row.get("proofType") or "").strip():
            failures.append(f"{fid}: proofType missing")

    return failures, by_id


def check_paths_exist(root: Path, rel_paths: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for rel in rel_paths:
        p = root / rel
        if not p.exists():
            missing.append(str(p))
    return missing


def run_cmd(cmd: List[str], cwd: Path) -> CommandResult:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return CommandResult(cmd=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def summarize_result(result: CommandResult) -> Dict:
    return {
        "cmd": result.cmd,
        "returncode": result.returncode,
        "ok": result.ok,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-40:]),
    }

