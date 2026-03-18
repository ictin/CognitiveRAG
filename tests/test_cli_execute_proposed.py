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


def test_execute_proposed_with_content(tmp_path):
    # prepare content file and pass execute flag
    content = tmp_path / 'content.txt'
    target = './example_exec.py'
    content.write_text('from __future__ import annotations\nprint("exec-ok")\n')
    args = ['analyze', '--text', f'from __future__ import annotations {target}', '--cwd', str(REPO_ROOT), '--execute-proposed-repair', '--content-file', str(content)]
    rc, out, err = run_module(args)
    assert rc == 0
    j = json.loads(out)
    assert j['executed_repair'] is True
    # execution_result should contain workflow_result with prefix_ok true
    er = j.get('execution_result')
    assert er is not None
    wf = er.get('workflow_result')
    assert wf.get('workflow_ok') is True
    assert wf.get('fix_result', {}).get('last_verify', {}).get('prefix_ok') is True
