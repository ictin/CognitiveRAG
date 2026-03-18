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


def test_analyze_syntax_error():
    args = ['analyze', '--text', 'SyntaxError: invalid syntax in file bad.py', '--cwd', str(REPO_ROOT)]
    rc, out, err = run_module(args)
    assert rc == 0
    j = json.loads(out)
    cls = j['classification']
    assert cls['syntax_error'] is True
    assert j['deterministic_repair_recommended'] is True


def test_analyze_import_error():
    args = ['analyze', '--text', "ModuleNotFoundError: No module named 'foo'", '--cwd', str(REPO_ROOT)]
    rc, out, err = run_module(args)
    assert rc == 0
    j = json.loads(out)
    cls = j['classification']
    assert cls['import_error'] is True
    # without exact-bytes risk, recommendation may be False
    assert isinstance(j['deterministic_repair_recommended'], bool)


def test_analyze_exact_bytes():
    # include a target token so a proposed command can be generated
    args = ['analyze', '--text', 'from __future__ import annotations ./example.py', '--cwd', str(REPO_ROOT)]
    rc, out, err = run_module(args)
    assert rc == 0
    j = json.loads(out)
    cls = j['classification']
    assert cls['exact_bytes_risk'] is True
    assert j['deterministic_repair_recommended'] is True
    # proposed_repair_command should be present when a target is inferred
    assert j.get('proposed_repair_command') is not None


def test_analyze_benign():
    args = ['analyze', '--text', 'informational log: started', '--cwd', str(REPO_ROOT)]
    rc, out, err = run_module(args)
    assert rc == 0
    j = json.loads(out)
    cls = j['classification']
    assert cls['syntax_error'] is False
    assert cls['import_error'] is False
    assert cls['py_compile_failure'] is False
    assert cls['pytest_failure'] is False
    assert cls['exact_bytes_risk'] is False
    assert j['deterministic_repair_recommended'] is False
    # executing without content_file should skip with a reason
    args2 = ['analyze', '--text', 'from __future__ import annotations ./example.py', '--cwd', str(REPO_ROOT), '--execute-proposed-repair']
    rc2, out2, err2 = run_module(args2)
    assert rc2 == 0
    j2 = json.loads(out2)
    assert j2['executed_repair'] is False
    assert j2['execution_skipped_reason'] is not None
