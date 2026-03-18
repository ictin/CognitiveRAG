import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(REPO_ROOT / '.venv' / 'bin' / 'python')
MODULE = 'CognitiveRAG.cli'


def run_module(args):
    proc = subprocess.Popen([PYTHON, '-m', MODULE] + args, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def test_module_repair_success(tmp_path):
    content = tmp_path / 'c.txt'
    target = tmp_path / 't.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    args = ['repair', '--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--py-compile']
    rc, out, err = run_module(args)
    assert rc == 0
    parsed = json.loads(out)
    wf = parsed.get('workflow_result')
    assert wf.get('workflow_ok') is True


def test_module_repair_mismatch(tmp_path):
    content = tmp_path / 'c2.txt'
    target = tmp_path / 't2.py'
    content.write_text('print("no future")\n')
    args = ['repair', '--path', str(target), '--content-file', str(content), '--expected-prefix', 'from __future__ import annotations', '--py-compile']
    rc, out, err = run_module(args)
    assert rc == 0
    parsed = json.loads(out)
    wf = parsed.get('workflow_result')
    assert wf.get('workflow_ok') is False
