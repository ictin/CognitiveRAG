AUTONOMOUS_REPAIR_LOOP

Trigger conditions

- Exact syntax errors (SyntaxError, indentation errors)
- Broken imports (ModuleNotFoundError, ImportError)
- Failing py_compile for a file
- Failing pytest when failures are caused by file content
- Any case where exact bytes matter (shebangs, __future__ imports, encoded literals)

Decision rules

- Prefer deterministic repair workflow when the failure requires exact-syntax or verified on-disk bytes.
- Use direct in-memory edits only for non-exact, safe refactors (typo fixes that don't change syntax-sensitive tokens).
- Require content-file based repair when:
  - the file needs exact bytes (double underscores, special whitespace)
  - the change is risky to perform with in-memory string transformations
- Stop and ask for human input when:
  - the target is large or multi-file and repair requires broader design decisions
  - repeated retries fail or tests indicate logic-level regressions
- Retry automatically once on prefix mismatch or mechanical encoding issues; escalate after a second failure.

Standard autonomous loop

1. Detect failure (py_compile error, pytest failure, import error).
2. Identify target file and failing region.
3. Generate exact content to replace the target file (content file), preserving required bytes.
4. Write the content file locally (temp file).
5. Call the deterministic repair workflow:
   ./.venv/bin/python -m CognitiveRAG.cli repair --path <target> --content-file <content> --expected-prefix "<prefix>" --cwd /home/ictin_claw/.openclaw/workspace/CognitiveRAG [--py-compile] [--targeted-pytest <tests...>]
6. Inspect JSON result.
7. If workflow_ok == false:
   - If fix_result indicates a prefix_mismatch and the change is mechanical, retry once.
   - Otherwise stop and report to human with diagnostics (stdout/stderr, sha256, first_lines_repr, trace)

Validation requirements

- Always verify the following after a repair:
  - prefix_ok is true
  - first_lines_repr contains the expected prefix
  - sha256 matches the on-disk file
  - py_compile returncode == 0 if requested
  - pytest returncodes == 0 for targeted or full runs when requested

Safety rules

- Do not modify unrelated files.
- Prefer one-file repairs first; avoid wide refactors without human approval.
- Do not claim success without verification of on-disk content and test outcomes.
- Never use plain pytest; always run via ./.venv/bin/python -m pytest
- Never rely on implicit formatting or editor transforms; treat code as raw bytes when required.

Example autonomous scenarios

- Fixing a broken __future__ import:
  - Detect SyntaxError or py_compile failure referencing the file
  - Generate file content starting with: from __future__ import annotations
  - Run the repair workflow and verify prefix_ok and py_compile

- Fixing a malformed import line:
  - Detect ModuleNotFoundError
  - Adjust the import line content, write a content file, run repair workflow, run targeted pytest

- Repairing a test file and running targeted pytest:
  - When a single test fails due to a broken test file, repair the file and run targeted pytest via the workflow

- Escalation:
  - If retry fails or tests continue to fail, stop and notify a human operator with full diagnostics and suggested next steps

Constraints

- Create only this file.
- Do not modify any code as part of this document.
