"""Self-heal smoke test for CognitiveRAG.

Creates a temporary file, writes content via fix_and_verify_file, and runs a
harmless command to validate the environment.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import importlib.util

# Import self_heal by file to avoid importing the top-level package and its heavy dependencies
def _load_self_heal(repo_root: Path):
    mod_path = repo_root / 'agent' / 'self_heal.py'
    spec = importlib.util.spec_from_file_location('cognitive_self_heal', str(mod_path))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    self_heal = _load_self_heal(repo_root)
    repo_root = Path(__file__).resolve().parents[1]
    # Create a temporary directory under system temp
    with tempfile.TemporaryDirectory(prefix="self_heal_smoke_") as td:
        tdpath = Path(td)
        target = tdpath / "temp_target.py"
        content = "from __future__ import annotations\nprint('smoke')\n"
        # Use relative path inside cwd to exercise typical usage
        rel_target = str(target)
        fix_result = self_heal.fix_and_verify_file(path=rel_target, content=content, expected_prefix="from __future__ import annotations", cwd=str(repo_root))

        # Run harmless command
        repo_python = str(repo_root / ".venv" / "bin" / "python")
        cmd = [repo_python, "-c", "print('ok')"]
        command_result = self_heal.run_command(cmd, cwd=str(repo_root))

        smoke_passed = True
        lv = fix_result.get("last_verify")
        if not lv:
            smoke_passed = False
        else:
            smoke_passed = bool(lv.get("prefix_ok") and lv.get("sha256"))

        out = {
            "temp_file": str(target),
            "fix_result": fix_result,
            "command_result": command_result,
            "smoke_test_passed": smoke_passed,
        }

        print(json.dumps(out))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
