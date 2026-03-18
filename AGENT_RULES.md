File Operation Rules

- Always use literal overwrite (heredoc or bytes) for exact syntax files
- Never modify files incrementally when exact syntax is required
- Always read the file back from disk after writing
- Always print:
  - absolute path
  - first N lines with repr()
  - sha256

Execution Rules

- Always run from:
  /home/ictin_claw/.openclaw/workspace/CognitiveRAG
- Always use:
  ./.venv/bin/python
- Always run pytest as:
  python -m pytest

Validation Rules

- Never claim success without verification
- Always run:
  - py_compile
  - targeted pytest
  - full pytest
- If mismatch occurs -> retry automatically

Anti-Corruption Rules

- Never transform underscores (__future__)
- Never apply markdown formatting to code
- Never rewrite syntax implicitly
- Treat code as raw bytes when required

Failure Handling

- If environment mismatch:
  - detect interpreter
  - print diagnostics
- If file content mismatch:
  - rewrite atomically
  - re-verify

Output Rules

- Return results in code blocks only when executing
- No summaries when verification is requested
- No assumptions without proof
