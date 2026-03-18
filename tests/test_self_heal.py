import os
import json
import hashlib
import tempfile
import subprocess
from pathlib import Path

import pytest

import importlib.util
from pathlib import Path

# load self_heal by path to avoid importing top-level package and heavy deps
repo_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('cognitive_self_heal', str(repo_root / 'agent' / 'self_heal.py'))
_self_heal = importlib.util.module_from_spec(_spec)
_loader = _spec.loader
assert _loader is not None
_loader.exec_module(_self_heal)
self_heal = _self_heal


def test_sha256_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    h = self_heal.sha256_file(str(p))
    assert isinstance(h, str) and len(h) == 64


def test_read_first_lines_repr(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("line1\nline2\nline3\n")
    r = self_heal.read_first_lines_repr(str(p), n=2)
    assert "line1" in r and "line2" in r


def test_atomic_write_text_and_verify(tmp_path):
    p = tmp_path / "c.py"
    content = "from __future__ import annotations\nprint(\"hi\")\n"
    self_heal.atomic_write_text(str(p), content)
    assert p.exists()
    v = self_heal.verify_file_contents(str(p), expected_prefix="from __future__ import annotations")
    assert v.get("prefix_ok") is True
    assert v.get("sha256")


def test_verify_prefix_mismatch(tmp_path):
    p = tmp_path / "d.py"
    p.write_text("print('no future')\n")
    v = self_heal.verify_file_contents(str(p), expected_prefix="from __future__ import annotations")
    assert v.get("prefix_ok") is False


def test_run_command_ok(repo_root=None):
    repo_root = repo_root or Path.cwd()
    python = str(Path(repo_root) / ".venv" / "bin" / "python")
    res = self_heal.run_command([python, "-c", "print('ok')"], cwd=str(repo_root))
    assert res.get("returncode") == 0
    assert "ok" in res.get("stdout")


def test_fix_and_verify_success(tmp_path):
    target = tmp_path / "e.py"
    content = "from __future__ import annotations\nprint('x')\n"
    res = self_heal.fix_and_verify_file(str(target), content, expected_prefix="from __future__ import annotations", cwd=str(Path.cwd()))
    assert res.get("last_verify") and res.get("last_verify")["prefix_ok"] is True
    assert res.get("attempts") == 1


def test_fix_and_verify_retry_on_mismatch(tmp_path):
    target = tmp_path / "f.py"
    content = "print('no future')\n"
    res = self_heal.fix_and_verify_file(str(target), content, expected_prefix="from __future__ import annotations", cwd=str(Path.cwd()))
    assert res.get("attempts") == 2
    assert res.get("last_verify") and res.get("last_verify")["prefix_ok"] is False
