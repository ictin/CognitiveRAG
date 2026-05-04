# Accepted Feature Dual-Surface Regression Gate

This gate enforces automatic coverage for accepted implemented features `F-001` through `F-020` across:

1. CognitiveRAG standalone backend
2. OpenClaw CognitiveRAG integration/plugin

## Commands

- Backend gate:
  - `PYTHONPATH=tools .venv/bin/python tools/run_accepted_feature_backend_gate.py`
- Plugin gate:
  - `PYTHONPATH=tools .venv/bin/python tools/run_accepted_feature_plugin_gate.py`
- Combined dual-surface gate:
  - `PYTHONPATH=tools .venv/bin/python tools/run_accepted_feature_dual_surface_gate.py`

## Reports

Each run writes a machine-readable report under `forensics/<timestamp>_*`:

- `backend_gate_report.json`
- `plugin_gate_report.json`
- `combined_gate_report.json`

The combined report fails closed when:

- a feature mapping is missing (`F-001`..`F-020`)
- mapped test files are missing
- backend gate fails
- plugin gate fails
