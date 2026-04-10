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

reset_state() {
  rm -rf "$REPO_ROOT/data/session_memory" || true
  mkdir -p "$REPO_ROOT/data/session_memory"
}

run_seq() {
  local label="$1"
  shift
  reset_state
  echo "==> ${label}"
  "$PYTHON_BIN" -m pytest -q "$@"
}

run_seq "Epic C bootstrap: selector metrics surface" \
  CognitiveRAG/tests/context_selection/test_selector_metrics_audit.py

run_seq "Epic C bootstrap: C2/C3 latency+stability benchmarks" \
  CognitiveRAG/tests/benchmarks/test_c2_c3_benchmarks.py

run_seq "Epic C bootstrap: fast retrieval benchmark surface" \
  CognitiveRAG/tests/benchmarks/test_i_fast_retrieval_benchmark.py

echo "Epic C bootstrap sequential checks complete."
