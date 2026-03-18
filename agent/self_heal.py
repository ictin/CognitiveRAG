"""self_heal - small helper utilities to enforce AGENT_RULES.md for file fixes.

This module provides utility functions to compute file SHA256, read first
lines with repr(), atomically write files, verify contents, run shell
commands, and a convenience function to write-and-verify with a single
retry on mismatch.

Only uses Python standard library so it can be executed in isolated
venvs without external deps.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List


def sha256_file(path: str) -> str:
    """Return the SHA256 hex digest of the file at path.

    Args:
        path: Path to the file.

    Returns:
        Hexadecimal SHA256 digest string.
    """
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_first_lines_repr(path: str, n: int = 5) -> str:
    """Read the first n lines of a text file and return a joined repr() string.

    Decoding uses UTF-8 with replacement for invalid bytes so the output is
    stable and safe to display.

    Args:
        path: Path to the file.
        n: Number of lines to read.

    Returns:
        A single string containing repr() of each of the first n lines joined by '\n'.
    """
    p = Path(path)
    lines: List[str] = []
    with p.open("rb") as f:
        for _ in range(n):
            b = f.readline()
            if not b:
                break
            lines.append(repr(b.decode("utf-8", errors="replace")))
    return "\n".join(lines)


def atomic_write_text(path: str, content: str) -> None:
    """Atomically write text content to path using a temporary file and replace.

    The temporary file is created next to the destination to ensure a
    same-filesystem rename, then os.replace() is used for atomicity.

    Args:
        path: Destination file path.
        content: Text content to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-self-heal-", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as tf:
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_path, p)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def verify_file_contents(path: str, expected_prefix: str) -> Dict[str, object]:
    """Verify on-disk file contents and return structured info.

    Args:
        path: Path to the file to verify.
        expected_prefix: Expected starting string (text). The check uses direct
                         string prefix comparison on the decoded UTF-8 file
                         contents (errors replaced).

    Returns:
        A dictionary with keys: absolute_path, first_lines_repr, sha256, prefix_ok
    """
    p = Path(path)
    abs_path = str(p.resolve())
    first_lines = read_first_lines_repr(abs_path, n=5)
    sha = sha256_file(abs_path)
    # Read full decoded content for prefix check but avoid huge memory by reading only len(expected_prefix)+1 bytes
    prefix_ok = False
    if expected_prefix is None:
        prefix_ok = True
    else:
        expected_len = len(expected_prefix)
        with p.open("rb") as f:
            data = f.read(expected_len + 1)
        decoded = data.decode("utf-8", errors="replace")
        prefix_ok = decoded.startswith(expected_prefix)
    return {
        "absolute_path": abs_path,
        "first_lines_repr": first_lines,
        "sha256": sha,
        "prefix_ok": prefix_ok,
    }


def run_command(cmd: List[str], cwd: str) -> Dict[str, object]:
    """Run a command and capture return code, stdout, and stderr.

    Args:
        cmd: List of command arguments.
        cwd: Working directory to run the command in.

    Returns:
        A dictionary with keys: command, returncode, stdout, stderr
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": out,
        "stderr": err,
    }


def fix_and_verify_file(path: str, content: str, expected_prefix: str, cwd: str) -> Dict[str, object]:
    """Atomically write content, verify on-disk, retry once on mismatch.

    Behavior:
    - Write atomically using atomic_write_text
    - Verify via verify_file_contents
    - If prefix_ok is False, retry write+verify once more

    Returns a dictionary with keys:
    - attempts: int
    - last_verify: result of verify_file_contents
    - actions: list of actions taken (strings)
    """
    actions: List[str] = []
    attempts = 0
    last_verify = None
    for attempt in range(2):
        attempts += 1
        actions.append(f"atomic_write_attempt_{attempt + 1}")
        atomic_write_text(path, content)
        actions.append("read_back")
        last_verify = verify_file_contents(path, expected_prefix)
        if last_verify.get("prefix_ok"):
            actions.append("verified_ok")
            break
        else:
            actions.append("prefix_mismatch")
    return {"attempts": attempts, "last_verify": last_verify, "actions": actions}


def run_repair_workflow(path: str, content_file: str, expected_prefix: str, cwd: str, py_compile: bool = False, targeted_pytest: list | None = None, full_pytest: bool = False) -> Dict[str, object]:
    """Invoke the deterministic repair workflow (tools/run_repair_workflow.py) and return parsed JSON output.

    Args:
        path: target file path
        content_file: path to the content file to write
        expected_prefix: expected prefix string
        cwd: repository working directory
        py_compile: whether to run py_compile
        targeted_pytest: list of test paths for targeted pytest
        full_pytest: whether to run full pytest

    Returns:
        Parsed JSON object returned by the workflow runner, or a dict with error details on failure.
    """
    repo_python = str(Path(cwd) / ".venv" / "bin" / "python")
    workflow_script = str(Path(cwd) / "tools" / "run_repair_workflow.py")
    cmd = [repo_python, workflow_script, "--path", path, "--content-file", content_file, "--expected-prefix", expected_prefix, "--cwd", cwd]
    if py_compile:
        cmd.append("--py-compile")
    if targeted_pytest:
        cmd.append("--targeted-pytest")
        cmd.extend(list(targeted_pytest))
    if full_pytest:
        cmd.append("--full-pytest")
    res = run_command(cmd, cwd=cwd)
    out = res.get("stdout", "")
    try:
        parsed = json.loads(out)
        return {"command_result": res, "workflow_result": parsed}
    except Exception as e:
        return {"command_result": res, "error": repr(e), "stdout": out, "stderr": res.get("stderr")}
