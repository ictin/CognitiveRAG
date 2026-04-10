#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHONPATH="./CognitiveRAG"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

cleanup_runtime_state() {
  rm -rf "$REPO_ROOT/data/session_memory" || true
  mkdir -p "$REPO_ROOT/data/session_memory"
}

run_seq() {
  local label="$1"
  shift
  cleanup_runtime_state
  echo "==> ${label}"
  "$PYTHON_BIN" -m pytest -q "$@"
}

run_seq "B3 contradiction/compatibility audit" \
  CognitiveRAG/tests/context_selection/test_contradiction_compatibility_audit.py

run_seq "B3 assemble integration" \
  CognitiveRAG/tests/context_selection/test_assemble_integration.py

run_seq "B3 selector metrics audit" \
  CognitiveRAG/tests/context_selection/test_selector_metrics_audit.py

run_seq "B3 retrieval integration slice" \
  CognitiveRAG/tests/retrieval/test_router_selector_integration.py

echo "B3 sequential validation complete."
