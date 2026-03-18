import importlib.util
from pathlib import Path

import pytest

# load module by path
repo_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('repair_decision', str(repo_root / 'agent' / 'repair_decision.py'))
_rd = importlib.util.module_from_spec(_spec)
_loader = _spec.loader
assert _loader is not None
_loader.exec_module(_rd)
repair_decision = _rd


def test_classify_syntax_and_indentation(tmp_path):
    text1 = "SyntaxError: invalid syntax in file bad.py"
    c1 = repair_decision.classify_failure(text1)
    assert c1['syntax_error'] is True

    text2 = "IndentationError: unexpected indent"
    c2 = repair_decision.classify_failure(text2)
    assert c2['syntax_error'] is True


def test_classify_import_and_py_compile_and_pytest():
    text = "ModuleNotFoundError: No module named 'foo' in app.py"
    c = repair_decision.classify_failure(text)
    assert c['import_error'] is True

    text_py = "py_compile: failed due to invalid syntax"
    cpy = repair_decision.classify_failure(text_py)
    assert cpy['py_compile_failure'] is True

    text_pt = "================================== FAILURES ===================================\npytest reported failures"
    cpt = repair_decision.classify_failure(text_pt)
    assert cpt['pytest_failure'] is True


def test_exact_bytes_risk_detection():
    texts = [
        "from __future__ import annotations in header",
        "#! /usr/bin/env python3 with shebang",
        "# -*- coding: utf-8 -*- encoding marker",
        "malformed import line: frommodule import x",
    ]
    for t in texts:
        c = repair_decision.classify_failure(t)
        assert c['exact_bytes_risk'] is True


def test_benign_text():
    t = "some informational log: started processing"
    c = repair_decision.classify_failure(t)
    assert c['syntax_error'] is False
    assert c['import_error'] is False
    assert c['py_compile_failure'] is False
    assert c['pytest_failure'] is False
    assert c['exact_bytes_risk'] is False


def test_should_use_deterministic_repair():
    assert repair_decision.should_use_deterministic_repair({'syntax_error': True}) is True
    assert repair_decision.should_use_deterministic_repair({'py_compile_failure': True}) is True
    assert repair_decision.should_use_deterministic_repair({'exact_bytes_risk': True}) is True
    assert repair_decision.should_use_deterministic_repair({'pytest_failure': True}) is False
    assert repair_decision.should_use_deterministic_repair({}) is False


def test_should_retry_mechanical_mismatch():
    wf = {'workflow_ok': False, 'fix_result': {'actions': ['atomic_write_attempt_1', 'read_back', 'prefix_mismatch']}}
    assert repair_decision.should_retry_mechanical_mismatch(wf) is True

    wf2 = {'workflow_ok': True, 'fix_result': {'actions': []}}
    assert repair_decision.should_retry_mechanical_mismatch(wf2) is False

    assert repair_decision.should_retry_mechanical_mismatch({}) is False
