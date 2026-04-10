# CognitiveRAG

Backend intelligence layer for OpenClaw memory, retrieval, and context construction.

## What CognitiveRAG Is

CognitiveRAG is a multi-layer memory and context-construction backend for OpenClaw. It builds typed evidence candidates across retrieval lanes, applies policy/scoring/selection constraints, and returns explanation artifacts that show why context was selected.

## Why This Exists

OpenClaw needs a backend that can do more than simple top-k recall. CognitiveRAG exists to:
- remember important information over time
- retrieve the best current evidence across local and web-aware lanes
- preserve reusable conclusions and promoted memory
- support bounded discovery and deeper follow-up when useful
- explain selection decisions so behavior is inspectable and testable

## Core Guarantees

CognitiveRAG is designed to keep these guarantees explicit:
- backend owns intelligence and canonical retrieval/memory policy
- source-class and truthfulness boundaries are preserved in outputs
- context selection is constrained by policy and budget, not unlimited append
- explanation artifacts are first-class outputs, not optional debugging text

## Current Architecture

Primary backend surfaces in this repo:
- API/service entrypoints: `CognitiveRAG/main_server.py`, `CognitiveRAG/api/routes/*`
- Retrieval lanes and routing: `CognitiveRAG/crag/retrieval/*`
- Context selection and explanation model: `CognitiveRAG/crag/context_selection/*`
- Session memory and compaction/recovery: `CognitiveRAG/session_memory/*`
- Promoted/reasoning/web/skill memory subsystems: `CognitiveRAG/memory/*`, `CognitiveRAG/web_memory/*`, `CognitiveRAG/skill_memory/*`

## Memory Taxonomy

Current code paths cover:
- session/episodic memory
- promoted memory
- reasoning memory
- skill execution/evaluation memory
- web evidence and promoted web memory
- corpus/lexical/semantic retrieval-backed memory access

Markdown mirror outputs are integration artifacts, not the full memory system.

## Context Construction Model

High-level model used by current backend path:
1. build typed candidates from retrieval/memory lanes
2. perform lane-local pruning and normalization
3. compute utility and policy-conditioned scoring signals
4. apply constrained selection under token/budget rules
5. reorder for long-context utility where applicable
6. emit explanation artifacts describing selected context and influence

## What Is Already Implemented

Implemented baseline includes:
- multi-lane retrieval foundation
- context-selection foundation with explanation outputs
- promoted memory and reasoning memory slices
- web evidence + promoted web slices
- bounded discovery/controller slices
- skill-memory execution/evaluation slices

## What Is Partial

Current Epic B parity status:
- B1 typed candidate coverage: `PARTLY_BUILT`
- B2 scoring and token-budget rules: `PARTLY_BUILT`
- B3 contradiction/compatibility filtering: `PARTLY_BUILT`
- B4 reorder and explanation output: `FULLY_BUILT`

So parity work remains active on B1-B3.

## What Is Not Started Yet (Current Program Scope)

- Graph-phase delivery as an active implementation phase is not started in the current program order.
- Current focus remains Epic B parity, then Epic C safety/metrics.

## Repo Role In OpenClaw Ecosystem

- `CognitiveRAG` (this repo): backend intelligence and canonical truth
- `openclaw-cognitiverag-memory`: OpenClaw integration adapter
- `openclaw-upstream`: OpenClaw runtime/core (not changed unless proven unavoidable)

## Setup / Run / Test

From repo root:
```bash
cd /home/ictin_claw/.openclaw/workspace/CognitiveRAG
```

Install dependencies (typical local workflow):
```bash
python3 -m pip install -r CognitiveRAG/requirements.txt
```

Run backend service (example):
```bash
python3 -m uvicorn CognitiveRAG.main_server:app --reload --host 127.0.0.1 --port 8080
```

Run tests (targeted first):
```bash
python3 -m pytest CognitiveRAG/tests/context_selection -q
python3 -m pytest CognitiveRAG/tests/retrieval -q
python3 -m pytest CognitiveRAG/tests -q
```

## Current Roadmap Phase

Current order:
1. Epic A completed
2. Epic B (current): context design/code parity audit and gap closure
3. Epic C: metrics, smoke, and regression safety
4. Graph layer later

Graph is planned, but it is not presented as current-phase completed work.
