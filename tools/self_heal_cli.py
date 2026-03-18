"""Command-line wrapper for CognitiveRAG.agent.self_heal.

Reads content from a file, invokes fix_and_verify_file, optionally runs
py_compile and pytest using the repository venv, and prints a single JSON
object with results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import importlib.util

try:
    from CognitiveRAG.agent import self_heal
except Exception:
    # Fallback: load the module by path so the script works when executed from repo root
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cognitive_self_heal", str(repo_root / "agent" / "self_heal.py"))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    self_heal = module


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser("self_heal_cli")
    parser.add_argument("--path", required=True, help="Path to target file to write/verify")
    parser.add_argument("--content-file", required=True, help="File containing the content to write")
    parser.add_argument("--expected-prefix", required=False, default=None, help="Expected prefix to verify")
    parser.add_argument("--cwd", required=False, default=os.getcwd(), help="Working directory to run commands in")
    parser.add_argument("--run-py-compile", action="store_true", help="Run py_compile on the target path")
    parser.add_argument("--run-targeted-pytest", nargs="*", default=[], help="List of test paths to run as targeted pytest")
    parser.add_argument("--run-full-pytest", action="store_true", help="Run full pytest (tests)")

    args = parser.parse_args(argv)

    content_path = Path(args.content_file)
    if not content_path.exists():
        print(json.dumps({"error": f"content file not found: {str(content_path)}"}))
        return 2

    content = content_path.read_text(encoding="utf-8")

    fix_result = self_heal.fix_and_verify_file(
        path=args.path, content=content, expected_prefix=args.expected_prefix, cwd=args.cwd
    )

    output = {"fix_result": fix_result}

    # Run py_compile if requested
    repo_python = str(Path(args.cwd) / ".venv" / "bin" / "python")

    if args.run_py_compile:
        target_rel = args.path
        # If args.path is absolute and inside cwd, make it relative
        try:
            p = Path(args.path)
            cwdp = Path(args.cwd)
            if p.is_absolute() and str(p).startswith(str(cwdp)):
                rel = str(p.relative_to(cwdp))
                target_rel = rel
        except Exception:
            target_rel = args.path
        cmd = [repo_python, "-m", "py_compile", target_rel]
        output["py_compile_result"] = self_heal.run_command(cmd, cwd=args.cwd)

    if args.run_targeted_pytest:
        cmd = [repo_python, "-m", "pytest"] + list(args.run_targeted_pytest)
        output["targeted_pytest_result"] = self_heal.run_command(cmd, cwd=args.cwd)

    if args.run_full_pytest:
        cmd = [repo_python, "-m", "pytest", "tests"]
        output["full_pytest_result"] = self_heal.run_command(cmd, cwd=args.cwd)

    # Print single JSON object
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
