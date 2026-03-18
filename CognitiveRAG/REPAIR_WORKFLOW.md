REPAIR_WORKFLOW

Preferred user-facing entrypoint

- use:
  ./.venv/bin/python -m CognitiveRAG.cli repair --path <target> --content-file <content> --expected-prefix "<prefix>" [--py-compile] [--targeted-pytest <tests...>] [--full-pytest]
- this is the default recommended path for deterministic repairs

When to use the repair workflow

- Exact syntax fixes where bytes matter (tests, __future__ imports, shebangs)
- Broken imports that require deterministic file edits
- Test-file repairs that must pass py_compile and pytest
- Any case where file content must be verified on disk before claiming success

Required working directory

- /home/ictin_claw/.openclaw/workspace/CognitiveRAG

Required interpreter

- ./.venv/bin/python

Standard workflow

1. Prepare a content file with the exact bytes to write. Use a literal heredoc or bytes-level write to avoid transformations.
2. Run the deterministic workflow (preferred CLI):
   ./.venv/bin/python -m CognitiveRAG.cli repair --path <target> --content-file <content> --expected-prefix "<prefix>" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG [--py-compile] [--targeted-pytest <tests...>] [--full-pytest]
   Lower-level options:
   - tools/run_repair_workflow.py (direct runner)
   - tools/self_heal_cli.py (alternative CLI wrapper)
   - agent/self_heal.py (programmatic helper)
3. Inspect the single JSON object printed to stdout.
4. Confirm:
   - workflow_ok == true
   - fix_result.last_verify.prefix_ok == true
   - fix_result.last_verify.first_lines_repr contains the expected prefix
   - fix_result.last_verify.sha256 is present
   - py_compile_result.returncode == 0 (if requested)
   - targeted_pytest_result.returncode == 0 (if requested)
   - full_pytest_result.returncode == 0 (if requested)

Exact command templates (CLI preferred)

- CLI repair + py_compile:
  ./.venv/bin/python -m CognitiveRAG.cli repair --path path/to/target.py --content-file /tmp/content.txt --expected-prefix "from __future__ import annotations" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG --py-compile

- CLI repair + targeted pytest:
  ./.venv/bin/python -m CognitiveRAG.cli repair --path path/to/target.py --content-file /tmp/content.txt --expected-prefix "from __future__ import annotations" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG --targeted-pytest tests/test_phase4_web_balance.py

- CLI repair + full pytest:
  ./.venv/bin/python -m CognitiveRAG.cli repair --path path/to/target.py --content-file /tmp/content.txt --expected-prefix "from __future__ import annotations" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG --full-pytest

- Future import case (exact bytes):
  Ensure the content file begins exactly with the following line (no substitutions):
  from __future__ import annotations

Rules

- Never claim success without on-disk verification (read back and sha256).
- Never use plain pytest — always run via the repo interpreter: ./.venv/bin/python -m pytest
- Never rely on implicit formatting or editor transforms when preparing content.
- Always verify first_lines_repr in the JSON output.
- Always verify sha256 in the JSON output.

Failure recovery

- Prefix mismatch:
  - Re-open the content file and confirm the first bytes match the expected prefix exactly.
  - Rewrite atomically using a literal/heredoc or bytes-level write and re-run the workflow.
- Missing venv python:
  - Verify .venv exists and contains bin/python. If not, re-create or activate the correct venv.
- CLI import issues (ModuleNotFoundError CognitiveRAG):
  - Ensure you run the CLI from the repo root.
  - The tools include fallback loaders that import agent/self_heal.py by path; if that fails, inspect PYTHONPATH and venv.
- Pytest failures after successful write:
  - Inspect pytest output in targeted_pytest_result or full_pytest_result.
  - If failures are unrelated to the repair, fix tests or mark them accordingly; if failures are caused by the change, revert and iterate.

Constraints

- Do not modify any other files.
- Create only this file when documenting the workflow.
