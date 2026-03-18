import json
import subprocess
from pathlib import Path
import tempfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(REPO_ROOT / '.venv' / 'bin' / 'python')
CLI = str(REPO_ROOT / 'tools' / 'run_repair_workflow.py')


def run_workflow(args, cwd=None):
    cwd = cwd or str(REPO_ROOT)
    proc = subprocess.Popen([PYTHON, CLI] + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def test_workflow_success_py_compile(tmp_path):
    content = tmp_path / 'c.txt'
    target = tmp_path / 't.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    args = ['--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--cwd', str(REPO_ROOT), '--py-compile']
    rc, out, err = run_workflow(args)
    assert rc == 0
    j = json.loads(out)
    assert j.get('workflow_ok') is True
    fr = j.get('fix_result')
    assert fr.get('last_verify', {}).get('prefix_ok') is True
    assert j.get('py_compile_result', {}).get('returncode') == 0


def test_workflow_prefix_mismatch(tmp_path):
    content = tmp_path / 'c2.txt'
    target = tmp_path / 't2.py'
    content.write_text('print("no future")\n')
    args = ['--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--cwd', str(REPO_ROOT), '--py-compile']
    rc, out, err = run_workflow(args)
    # rc may be 0; rely on JSON
    j = json.loads(out)
    assert j.get('workflow_ok') is False
    fr = j.get('fix_result')
    assert fr.get('attempts') == 2
    assert fr.get('last_verify', {}).get('prefix_ok') is False


def test_workflow_targeted_pytest(tmp_path):
    content = tmp_path / 'c3.txt'
    target = tmp_path / 't3.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    args = ['--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--cwd', str(REPO_ROOT), '--targeted-pytest', 'tests/test_self_heal.py']
    rc, out, err = run_workflow(args)
    assert rc == 0
    j = json.loads(out)
    assert 'targeted_pytest_result' in j
    assert j['targeted_pytest_result'].get('returncode') == 0
    assert j.get('workflow_ok') is True
