import json
import subprocess
from pathlib import Path
import tempfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(REPO_ROOT / '.venv' / 'bin' / 'python')
CLI = str(REPO_ROOT / 'tools' / 'self_heal_cli.py')


def run_cli(args, cwd=None):
    cwd = cwd or str(REPO_ROOT)
    proc = subprocess.Popen([PYTHON, CLI] + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def test_cli_success_with_future_prefix(tmp_path):
    content = tmp_path / 'c.txt'
    target = tmp_path / 't.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    args = ['--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--cwd', str(REPO_ROOT), '--run-py-compile']
    rc, out, err = run_cli(args)
    assert rc == 0
    j = json.loads(out)
    assert 'fix_result' in j
    lv = j['fix_result']['last_verify']
    assert lv['prefix_ok'] is True
    assert j.get('py_compile_result', {}).get('returncode') == 0


def test_cli_missing_content_file_returns_error(tmp_path):
    missing = tmp_path / 'nope.txt'
    target = tmp_path / 't2.py'
    args = ['--path', str(target), '--content-file', str(missing), '--expected-prefix', 'from __future__ import annotations', '--cwd', str(REPO_ROOT)]
    rc, out, err = run_cli(args)
    # Should exit non-zero or print structured error JSON
    assert rc != 0 or ('error' in out)


def test_cli_run_targeted_pytest_wiring(tmp_path):
    # Use existing small test to run as targeted pytest (tests/test_self_heal.py)
    content = tmp_path / 'c2.txt'
    target = tmp_path / 't3.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    args = ['--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--cwd', str(REPO_ROOT), '--run-targeted-pytest', 'tests/test_self_heal.py']
    rc, out, err = run_cli(args)
    # CLI should run and return rc 0 (pytest exitcode may be non-zero if tests fail, but in our case tests/test_self_heal.py should pass)
    assert rc == 0
    j = json.loads(out)
    assert 'targeted_pytest_result' in j


def test_cli_run_targeted_pytest_no_break(tmp_path):
    # Ensure targeted pytest argument can be empty list without crashing
    content = tmp_path / 'c3.txt'
    target = tmp_path / 't4.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    args = ['--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--cwd', str(REPO_ROOT), '--run-targeted-pytest']
    rc, out, err = run_cli(args)
    assert rc == 0
    # If no tests provided, CLI should still succeed and not include targeted_pytest_result
    j = json.loads(out)
    assert 'fix_result' in j
