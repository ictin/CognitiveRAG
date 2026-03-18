import json
from pathlib import Path

import pytest

import importlib.util

# load self_heal by path to avoid importing the package root
repo_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('cognitive_self_heal', str(repo_root / 'agent' / 'self_heal.py'))
_self_heal = importlib.util.module_from_spec(_spec)
_loader = _spec.loader
assert _loader is not None
_loader.exec_module(_self_heal)
self_heal = _self_heal


def test_wrapper_success(tmp_path):
    content = tmp_path / 'c.txt'
    target = tmp_path / 't.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    res = self_heal.run_repair_workflow(path=str(target), content_file=str(content), expected_prefix='from __future__ import annotations', cwd=str(repo_root), py_compile=True)
    cmd = res.get('command_result')
    wf = res.get('workflow_result')
    if wf is None:
        out = cmd.get('stdout', '')
        wf = json.loads(out)
    assert cmd.get('returncode') == 0
    assert wf.get('workflow_ok') is True
    assert wf.get('fix_result', {}).get('last_verify', {}).get('prefix_ok') is True
    assert wf.get('py_compile_result', {}).get('returncode') == 0


def test_wrapper_mismatch(tmp_path):
    content = tmp_path / 'c2.txt'
    target = tmp_path / 't2.py'
    content.write_text('print("no future")\n')
    res = self_heal.run_repair_workflow(path=str(target), content_file=str(content), expected_prefix='from __future__ import annotations', cwd=str(repo_root), py_compile=True)
    cmd = res.get('command_result')
    wf = res.get('workflow_result')
    if wf is None:
        out = cmd.get('stdout', '')
        wf = json.loads(out)
    assert cmd.get('returncode') == 0
    assert wf.get('workflow_ok') is False
    assert wf.get('fix_result', {}).get('attempts') == 2
    assert wf.get('fix_result', {}).get('last_verify', {}).get('prefix_ok') is False


def test_wrapper_targeted_pytest(tmp_path):
    content = tmp_path / 'c3.txt'
    target = tmp_path / 't3.py'
    content.write_text('from __future__ import annotations\nprint("ok")\n')
    res = self_heal.run_repair_workflow(path=str(target), content_file=str(content), expected_prefix='from __future__ import annotations', cwd=str(repo_root), targeted_pytest=['tests/test_self_heal.py'])
    cmd = res.get('command_result')
    wf = res.get('workflow_result')
    if wf is None:
        out = cmd.get('stdout', '')
        wf = json.loads(out)
    assert cmd.get('returncode') == 0
    assert wf.get('targeted_pytest_result', {}).get('returncode') == 0
    assert wf.get('workflow_ok') is True
