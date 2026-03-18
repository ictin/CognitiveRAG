"""Failure tests for self_heal utilities.

Runs a set of negative tests to ensure predictable failures and structured
diagnostics are returned rather than crashing the process.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
import tempfile


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
    tests = []
    all_expected = True

    # 1. verify_file_contents on non-existing file
    name = 'verify_nonexistent_file'
    try:
        path = str(repo_root / 'nonexistent_file_hopefully_nope.txt')
        try:
            res = self_heal.verify_file_contents(path, expected_prefix='x')
            passed = False
            details = {'result': res}
        except Exception as e:
            passed = True
            details = {'error': repr(e)}
    except Exception as e:
        passed = False
        details = {'error': repr(e)}
    tests.append({'name': name, 'passed': passed, 'details': details})
    all_expected = all_expected and passed

    # 2. run_command on non-existing executable
    name = 'run_nonexistent_executable'
    try:
        cmd = [str(repo_root / '.venv' / 'bin' / 'does-not-exist')]
        rc = self_heal.run_command(cmd, cwd=str(repo_root))
        # On some systems, Popen will return code !=0; treat non-zero as expected failure
        passed = rc.get('returncode') != 0
        details = rc
    except Exception as e:
        passed = True
        details = {'error': repr(e)}
    tests.append({'name': name, 'passed': passed, 'details': details})
    all_expected = all_expected and passed

    # 3. self_heal_cli with missing --content-file
    name = 'cli_missing_content_file'
    try:
        cli = repo_root / 'tools' / 'self_heal_cli.py'
        # Call python on the cli without providing content-file
        python = str(repo_root / '.venv' / 'bin' / 'python')
        proc = subprocess.Popen([python, str(cli), '--path', 'somepath'], cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        # Expect non-zero exit or JSON with error
        passed = proc.returncode != 0 or ('error' in out)
        details = {'returncode': proc.returncode, 'stdout': out, 'stderr': err}
    except Exception as e:
        passed = True
        details = {'error': repr(e)}
    tests.append({'name': name, 'passed': passed, 'details': details})
    all_expected = all_expected and passed

    # 4. fix_and_verify_file with expected_prefix that doesn't match
    name = 'fix_prefix_mismatch'
    try:
        with tempfile.TemporaryDirectory(prefix='self_heal_fail_') as td:
            target = Path(td) / 't.py'
            # content without the expected prefix
            content = "print('no prefix')\n"
            res = self_heal.fix_and_verify_file(path=str(target), content=content, expected_prefix='from __future__ import annotations', cwd=str(repo_root))
            lv = res.get('last_verify')
            passed = (res.get('attempts') == 2) and (lv is not None) and (lv.get('prefix_ok') is False)
            details = {'result': res}
    except Exception as e:
        passed = False
        details = {'error': repr(e)}
    tests.append({'name': name, 'passed': passed, 'details': details})
    all_expected = all_expected and passed

    out = {'tests': tests, 'all_expected_failures_observed': all_expected}
    print(json.dumps(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
