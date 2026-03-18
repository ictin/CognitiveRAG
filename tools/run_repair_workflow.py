"""Run a deterministic repair workflow using self-heal utilities.

Usage: run_repair_workflow.py --path <target> --content-file <file> --expected-prefix <prefix> --cwd <repo> [--py-compile] [--targeted-pytest tests...] [--full-pytest]

Prints a single JSON object with workflow_ok and step results.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import List, Optional


def load_self_heal(repo_root: Path):
    spec = importlib.util.spec_from_file_location("cognitive_self_heal", str(repo_root / "agent" / "self_heal.py"))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser("run_repair_workflow")
    parser.add_argument("--path", required=True)
    parser.add_argument("--content-file", required=True)
    parser.add_argument("--expected-prefix", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--py-compile", action="store_true")
    parser.add_argument("--targeted-pytest", nargs="*", default=[])
    parser.add_argument("--full-pytest", action="store_true")

    args = parser.parse_args(argv)

    repo_root = Path(args.cwd)
    self_heal = load_self_heal(repo_root)

    content = Path(args.content_file).read_text(encoding="utf-8")

    fix_result = self_heal.fix_and_verify_file(path=args.path, content=content, expected_prefix=args.expected_prefix, cwd=args.cwd)

    output = {"fix_result": fix_result}

    # prepare repo python
    repo_python = str(Path(args.cwd) / ".venv" / "bin" / "python")

    if args.py_compile:
        # make path relative if inside cwd
        target_rel = args.path
        try:
            p = Path(args.path)
            if p.is_absolute() and str(p).startswith(str(repo_root)):
                target_rel = str(p.relative_to(repo_root))
        except Exception:
            target_rel = args.path
        cmd = [repo_python, "-m", "py_compile", target_rel]
        output["py_compile_result"] = self_heal.run_command(cmd, cwd=args.cwd)

    if args.targeted_pytest:
        cmd = [repo_python, "-m", "pytest"] + list(args.targeted_pytest)
        output["targeted_pytest_result"] = self_heal.run_command(cmd, cwd=args.cwd)

    if args.full_pytest:
        cmd = [repo_python, "-m", "pytest", "tests"]
        output["full_pytest_result"] = self_heal.run_command(cmd, cwd=args.cwd)

    # determine workflow_ok
    workflow_ok = True
    lv = fix_result.get("last_verify")
    if not (lv and lv.get("prefix_ok")):
        workflow_ok = False
    if args.py_compile:
        if output.get("py_compile_result", {}).get("returncode") != 0:
            workflow_ok = False
    if args.targeted_pytest:
        if output.get("targeted_pytest_result", {}).get("returncode") != 0:
            workflow_ok = False
    if args.full_pytest:
        if output.get("full_pytest_result", {}).get("returncode") != 0:
            workflow_ok = False

    output = {"workflow_ok": workflow_ok, **output}
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
