import argparse
import importlib.util
import json
import os
from pathlib import Path

import uvicorn

from CognitiveRAG.app import create_app
from CognitiveRAG.core.settings import settings


def _load_self_heal(repo_root):
    spec = importlib.util.spec_from_file_location('cognitive_self_heal', str(repo_root / 'agent' / 'self_heal.py'))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(prog="cognitiverag")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default=settings.app.host)
    serve.add_argument("--port", type=int, default=settings.app.port)

    # repair subcommand: deterministic repair workflow for exact-syntax fixes
    repair = sub.add_parser("repair")
    repair.add_argument("--path", required=True)
    repair.add_argument("--content-file", required=True)
    repair.add_argument("--expected-prefix", required=True)
    repair.add_argument("--cwd", default="/home/ictin_claw/.openclaw/workspace/CognitiveRAG")
    repair.add_argument("--py-compile", action="store_true")
    repair.add_argument("--targeted-pytest", nargs="*", default=None)
    repair.add_argument("--full-pytest", action="store_true")

    # analyze subcommand
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--text", required=True)
    analyze.add_argument("--cwd", default="/home/ictin_claw/.openclaw/workspace/CognitiveRAG")
    analyze.add_argument("--execute-proposed-repair", action="store_true", help="Execute the proposed repair if safe and inputs provided")
    analyze.add_argument("--content-file", required=False, help="Content file to use if executing the proposed repair")
    analyze.add_argument("--expected-prefix", required=False, help="Expected prefix to use if executing the proposed repair")

    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run(
            "CognitiveRAG.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
    elif args.command == "repair":
        repo_root = Path(args.cwd)
        self_heal = _load_self_heal(repo_root)
        res = self_heal.run_repair_workflow(
            path=args.path,
            content_file=args.content_file,
            expected_prefix=args.expected_prefix,
            cwd=args.cwd,
            py_compile=args.py_compile,
            targeted_pytest=args.targeted_pytest,
            full_pytest=args.full_pytest,
        )
        # ensure valid JSON is printed for callers
        print(json.dumps(res))
    elif args.command == "analyze":
        # classify a failure text blob and recommend whether deterministic repair is preferred
        repo_root = Path(args.cwd)
        # load repair_decision module by path
        spec = importlib.util.spec_from_file_location('repair_decision', str(repo_root / 'agent' / 'repair_decision.py'))
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None
        loader.exec_module(module)
        rd = module
        classification = rd.classify_failure(args.text or "")
        recommended = rd.should_use_deterministic_repair(classification)
        proposed = None
        execution_result = None
        executed = False
        execution_skipped_reason = None
        # propose a repair command when recommended and a likely target exists
        target = classification.get("likely_repair_target")
        if recommended and target:
            # choose expected prefix heuristic
            if "__future__" in (args.text or "") or "from __future__" in (args.text or ""):
                expected = "from __future__ import annotations"
            else:
                expected = classification.get("reason") or ""
            cmd = (
                f"./.venv/bin/python -m CognitiveRAG.cli repair --path {target} --content-file <content> --expected-prefix \"{expected}\" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG --py-compile"
            )
            proposed = cmd

        # Optionally execute the proposed repair if requested and safe
        if args.execute_proposed_repair:
            if not recommended:
                execution_skipped_reason = "deterministic repair not recommended"
            elif not target:
                execution_skipped_reason = "no likely_repair_target inferred"
            elif not args.content_file:
                execution_skipped_reason = "no content_file provided"
            else:
                # need expected prefix: prefer explicit arg, else heuristic
                expected_to_use = args.expected_prefix or expected if 'expected' in locals() else None
                if not expected_to_use:
                    execution_skipped_reason = "no expected_prefix available"
                else:
                    # execute via self_heal.run_repair_workflow
                    self_heal = _load_self_heal(repo_root)
                    execution_result = self_heal.run_repair_workflow(path=target, content_file=args.content_file, expected_prefix=expected_to_use, cwd=args.cwd, py_compile=True)
                    executed = True
        out = {"classification": classification, "deterministic_repair_recommended": recommended, "proposed_repair_command": proposed, "executed_repair": executed, "execution_result": execution_result, "execution_skipped_reason": execution_skipped_reason}
        print(json.dumps(out))


if __name__ == "__main__":
    main()
