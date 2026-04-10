# CognitiveRAG

CognitiveRAG gives an OpenClaw agent better memory, better evidence selection, and clearer reasoning context for each turn.

## What CognitiveRAG Does For An OpenClaw Agent

CognitiveRAG is the backend intelligence layer that helps an agent:
- remember important facts without losing recoverability
- retrieve the best available evidence for the current turn (instead of dumping unranked context)
- reuse prior promoted knowledge and reasoning outcomes
- combine local corpus evidence with web evidence pathways
- explain why specific context was selected
- run bounded discovery loops when deeper exploration is useful

## Why This Is Different From Ordinary RAG

Ordinary RAG often stops at "retrieve top-k chunks and paste them." CognitiveRAG focuses on memory-aware, policy-constrained context construction:
- multi-lane candidate generation
- lane-local pruning and utility signals
- constrained selection under budget
- explanation artifacts as first-class output

This makes answers more auditable and better aligned to long-running agent workflows.

## Core Benefits

- Better continuity across sessions and turns
- Better evidence quality under token pressure
- Better reuse of validated/promoted knowledge
- Better transparency into context selection decisions
- Better safety around source-class and truthfulness boundaries

## Current Architecture

Primary backend surfaces:
- API/service entrypoints: `CognitiveRAG/main_server.py`, `CognitiveRAG/api/routes/*`
- Retrieval lanes and routing: `CognitiveRAG/crag/retrieval/*`
- Context selection and explanation model: `CognitiveRAG/crag/context_selection/*`
- Session memory compaction/recovery: `CognitiveRAG/session_memory/*`
- Promoted/reasoning/web/skill memory layers: `CognitiveRAG/memory/*`, `CognitiveRAG/web_memory/*`, `CognitiveRAG/skill_memory/*`

## Memory Taxonomy

Current memory/evidence layers include:
- session/episodic memory
- promoted memory
- reasoning memory
- skill execution/evaluation memory
- web evidence and promoted web memory
- corpus/lexical/semantic retrieval lanes

Markdown mirrors are integration artifacts, not the full memory system.

## Context Construction Model

Current backend flow:
1. build typed candidates across lanes
2. normalize and prune lane-local candidates
3. compute utility and policy-conditioned scoring signals
4. apply constrained selection under token/budget limits
5. reorder for long-context utility where relevant
6. emit explanation artifacts describing selected influence/context

## What Is Already Real

Implemented baseline includes:
- retrieval lanes and router foundation
- context-selection foundation with explanation outputs
- promoted memory and reasoning memory slices
- web evidence and promoted web slices
- bounded discovery/controller slices
- skill-memory execution/evaluation slices

## What Is Partial

Epic B parity status:
- B1 typed candidate coverage: `PARTLY_BUILT`
- B2 scoring and token-budget rules: `PARTLY_BUILT`
- B3 contradiction/compatibility filtering: `PARTLY_BUILT`
- B4 reorder and explanation output: `FULLY_BUILT`

NLI-style compatibility completeness should be treated as partial until Epic B parity closes it.

## What Is Next

Current order:
1. Epic A done
2. Epic B now (design/code parity closure)
3. Epic C next (metrics/smoke/regression safety)
4. Graph work later

Graph is planned, but not represented as current implementation.

## Repo Role In The OpenClaw Ecosystem

- `CognitiveRAG` (this repo): backend intelligence and canonical memory/retrieval/context logic
- `openclaw-cognitiverag-memory`: OpenClaw integration adapter
- `openclaw-upstream`: OpenClaw runtime/core (changed only if unavoidable and proven)

## Setup / Run / Test

From repo root:
```bash
cd /home/ictin_claw/.openclaw/workspace/CognitiveRAG
```

Install dependencies:
```bash
python3 -m pip install -r CognitiveRAG/requirements.txt
```

Run service (example):
```bash
python3 -m uvicorn CognitiveRAG.main_server:app --reload --host 127.0.0.1 --port 8080
```

Run tests:
```bash
python3 -m pytest CognitiveRAG/tests/context_selection -q
python3 -m pytest CognitiveRAG/tests/retrieval -q
python3 -m pytest CognitiveRAG/tests -q
```
