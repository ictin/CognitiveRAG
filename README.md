# CognitiveRAG

Backend intelligence layer for OpenClaw memory, retrieval, and context assembly.

## What CognitiveRAG Is

CognitiveRAG is the backend system that builds evidence candidates from multiple memory/retrieval lanes, scores and selects context under constraints, and returns explanation artifacts that make assembly decisions inspectable.

It is the canonical intelligence layer. The OpenClaw plugin is only the integration adapter.

## Why It Exists

OpenClaw needs a backend that can:
- unify multiple memory and evidence lanes
- select useful context under token limits
- preserve source-class/truthfulness boundaries
- produce inspectable decision artifacts

CognitiveRAG provides those backend responsibilities.

## Current Architecture

Major backend areas in this repository:
- API/server surfaces: `CognitiveRAG/main_server.py`, `CognitiveRAG/api/routes/*`
- Retrieval lanes and routing: `CognitiveRAG/crag/retrieval/*`
- Context selection and explanation path: `CognitiveRAG/crag/context_selection/*`
- Session memory and compaction: `CognitiveRAG/session_memory/*`
- Skill memory, promoted memory, and web evidence stores: `CognitiveRAG/skill_memory/*`, `CognitiveRAG/memory/*`, `CognitiveRAG/web_memory/*`

## Memory Taxonomy (Current)

The codebase currently distinguishes and uses:
- session/episodic memory
- promoted memory
- reasoning memory
- skill/execution memory
- web evidence and promoted web memory
- corpus/lexical/semantic retrieval lanes

## Current Implemented State (Truthful)

Status relative to the current Epic B parity audit baseline:
- B1 typed candidate coverage: `PARTLY_BUILT`
- B2 scoring and token-budget rules: `PARTLY_BUILT`
- B3 contradiction/compatibility filtering: `PARTLY_BUILT`
- B4 reorder and explanation output: `FULLY_BUILT`

Important clarity:
- compatibility/contradiction handling exists, but parity audit still treats it as partial
- global constrained selection behavior exists but parity remains partial versus canonical target design
- graph-first retrieval is **not** the current active phase of work

## What Is Partial vs Not Started

- Partial now: full parity on typed candidates, budget policy, and contradiction/compatibility filtering.
- Not started as current implementation focus: first graph layer delivery work (planned later phase, not current).

## Testing

This repository includes broad pytest coverage, including:
- context selection parity/audit tests
- retrieval lane and router tests
- skill memory and web memory tests
- session memory and lifecycle tests

Common commands:
```bash
cd /home/ictin_claw/.openclaw/workspace/CognitiveRAG
python -m pytest CognitiveRAG/tests/context_selection -q
python -m pytest CognitiveRAG/tests/retrieval -q
python -m pytest CognitiveRAG/tests -q
```

Use smaller targeted subsets first when auditing a specific Epic B story.

## Role in the OpenClaw Ecosystem

- `CognitiveRAG` (this repo): backend intelligence and canonical truth
- `openclaw-cognitiverag-memory`: OpenClaw integration adapter/plugin
- `openclaw-upstream`: OpenClaw core runtime (not modified unless unavoidable)

## Roadmap / Current Next Phase

Program order:
1. Epic A live signoff: complete
2. Epic B context design parity audit: current phase
3. Epic C metrics/smoke/regression safety
4. Epic E first graph layer

## Suggested GitHub Topics

`rag`, `memory-system`, `retrieval`, `context-selection`, `python`, `fastapi`, `openclaw`, `cognitiverag`
