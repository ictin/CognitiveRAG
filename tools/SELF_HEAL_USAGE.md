SELF_HEAL_USAGE

Purpose

- CognitiveRAG/agent/self_heal.py
  Provides small utilities used to safely fix repository files:
  - sha256_file(path) -> str
  - read_first_lines_repr(path, n=5) -> str
  - atomic_write_text(path, content) -> None
  - verify_file_contents(path, expected_prefix) -> dict
  - run_command(cmd, cwd) -> dict
  - fix_and_verify_file(path, content, expected_prefix, cwd) -> dict

- CognitiveRAG/tools/self_heal_cli.py
  A CLI wrapper that reads a content file, writes the target file using
  the self_heal utilities, and optionally runs py_compile and pytest. It
  prints a single JSON object with results.

- When to use
  Use these when you need to repair test files or other files where exact
  bytes/syntax matter and you want deterministic, verified changes.

Repo assumptions

- Run from:
  /home/ictin_claw/.openclaw/workspace/CognitiveRAG
- Use interpreter:
  ./.venv/bin/python

Typical workflow

1. Prepare the content file containing the exact bytes you want written.
2. Run the CLI with --content-file and --path pointing to the target.
3. Verify the JSON output (fix_result and optional py_compile/pytest results).
4. If targeted pytest is requested, inspect targeted_pytest_result.
5. Optionally run full pytest and inspect full_pytest_result.

Exact examples

- Fix a file from a content file (py_compile only):
  ./.venv/bin/python tools/self_heal_cli.py --path path/to/target.py --content-file /tmp/content.txt --expected-prefix "from __future__ import annotations" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG --run-py-compile

- Run targeted pytest after fix:
  ./.venv/bin/python tools/self_heal_cli.py --path path/to/target.py --content-file /tmp/content.txt --expected-prefix "from __future__ import annotations" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG --run-targeted-pytest tests/test_phase4_web_balance.py

- Run full pytest after fix:
  ./.venv/bin/python tools/self_heal_cli.py --path path/to/target.py --content-file /tmp/content.txt --expected-prefix "from __future__ import annotations" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG --run-full-pytest

- Handling the __future__ case safely:
  Ensure the file content begins with the exact bytes:
  from __future__ import annotations
  Do not transform or escape the double underscores.

Validation checklist

- JSON includes absolute_path
- JSON includes first_lines_repr containing the expected prefix
- JSON includes sha256
- prefix_ok is true
- If py_compile run: py_compile_result.returncode == 0
- If pytest run: pytest returncode == 0

Failure notes

- If prefix_ok is false:
  - Re-check the content file bytes for accidental transformations
  - Use atomic_write_text to write exact bytes and re-run

- If repo interpreter is missing:
  - Verify .venv exists and contains bin/python
  - If not, create or activ ate the venv used by the project

- If CLI import fails (ModuleNotFoundError CognitiveRAG):
  - Ensure you ran the CLI from the repo root (see Repo assumptions)
  - The CLI contains a fallback loader that loads agent/self_heal.py by path

- If pytest is not found:
  - Ensure .venv has pytest installed (./.venv/bin/pip install -r requirements-dev.txt or similar)

Constraints

- Do not modify any other files when repairing
- Use literal/heredoc or bytes-level writes for exact-syntax files
