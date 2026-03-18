import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(REPO_ROOT / '.venv' / 'bin' / 'python')
MODULE = 'CognitiveRAG.cli'


def run_cli(args):
    # Invoke CLI as a script file to avoid module import path issues
    cli_path = str(REPO_ROOT / 'CognitiveRAG' / 'cli.py')
    env = dict(**os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    proc = subprocess.Popen([PYTHON, cli_path] + args, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err = proc.communicate()
    return proc.returncode, out, err


def parse_output(out: str):
    try:
        return json.loads(out)
    except Exception:
        # fallback: try eval safely
        try:
            return eval(out, {})
        except Exception:
            return None


def test_cli_repair_success(tmp_path):
    content = tmp_path / 'c.txt'
    target = tmp_path / 't.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    args = ['repair', '--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--py-compile']
    rc, out, err = run_cli(args)
    assert rc == 0
    # require valid JSON on stdout
    parsed = json.loads(out)
    wf = parsed.get('workflow_result') if isinstance(parsed, dict) else parsed
    assert wf.get('workflow_ok') is True
    assert wf.get('fix_result', {}).get('last_verify', {}).get('prefix_ok') is True
    assert wf.get('py_compile_result', {}).get('returncode') == 0


def test_cli_repair_mismatch(tmp_path):
    content = tmp_path / 'c2.txt'
    target = tmp_path / 't2.py'
    content.write_text('print("no future")\n')
    args = ['repair', '--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--py-compile']
    rc, out, err = run_cli(args)
    # require valid JSON on stdout
    parsed = json.loads(out)
    wf = parsed.get('workflow_result') if isinstance(parsed, dict) else parsed
    assert wf.get('workflow_ok') is False
    assert wf.get('fix_result', {}).get('attempts') == 2
    assert wf.get('fix_result', {}).get('last_verify', {}).get('prefix_ok') is False


def test_cli_repair_targeted_pytest(tmp_path):
    content = tmp_path / 'c3.txt'
    target = tmp_path / 't3.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    args = ['repair', '--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--targeted-pytest', 'tests/test_self_heal.py']
    rc, out, err = run_cli(args)
    assert rc == 0
    # require valid JSON on stdout
    parsed = json.loads(out)
    wf = parsed.get('workflow_result') if isinstance(parsed, dict) else parsed
    assert wf.get('targeted_pytest_result', {}).get('returncode') == 0
    assert wf.get('workflow_ok') is True
