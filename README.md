# CognitiveRAG

CognitiveRAG is the intelligence backend that helps an OpenClaw agent remember what matters, choose better evidence, and explain its context decisions under real token and runtime constraints.

## What This Does For An OpenClaw Agent

CognitiveRAG improves agent behavior in ways users feel directly:
- better continuity across long sessions instead of fragile short-term chat memory
- better turn-level evidence selection instead of raw context stuffing
- better reuse of promoted knowledge and reasoning outcomes
- better integration of local corpus evidence and web evidence
- better explainability of why specific context was selected

## Why It Is Different From Ordinary RAG

Ordinary RAG often means top-k retrieval plus prompt append. CognitiveRAG is a memory-aware context-construction system:
- multiple retrieval/memory lanes feed typed candidates
- candidates are pruned/scored with policy and budget constraints
- selected context is reordered for long-context utility
- explanation artifacts are emitted as first-class outputs

## Core Benefits

- higher-quality context under token pressure
- stronger recoverability for important memory
- safer source-class and truthfulness boundaries
- reusable promoted knowledge instead of repeated rediscovery
- bounded discovery support for deeper investigative turns

## Current Architecture

Primary backend surfaces:
- API and routes: `CognitiveRAG/main_server.py`, `CognitiveRAG/api/routes/*`
- Retrieval lanes/router: `CognitiveRAG/crag/retrieval/*`
- Context selection/explanations: `CognitiveRAG/crag/context_selection/*`
- Session memory/compaction: `CognitiveRAG/session_memory/*`
- Promoted/reasoning/web/skill memory: `CognitiveRAG/memory/*`, `CognitiveRAG/web_memory/*`, `CognitiveRAG/skill_memory/*`

## Memory Taxonomy

Current layers include:
- session and episodic memory
- promoted memory
- reasoning memory
- skill execution/evaluation memory
- web evidence and promoted web memory
- corpus lexical/semantic retrieval lanes

Markdown mirrors are integration artifacts, not the full memory system.

## Current Implementation Status

Implemented baseline includes:
- retrieval lane foundation
- context-selection foundation with explanation output
- promoted and reasoning memory slices
- web evidence and promoted web slices
- bounded discovery/controller slices
- skill-memory execution/evaluation slices

Epic B parity status:
- B1 typed candidate coverage: `PARTLY_BUILT`
- B2 scoring and token-budget rules: `PARTLY_BUILT`
- B3 contradiction/compatibility filtering: `PARTLY_BUILT`
- B4 reorder and explanation output: `FULLY_BUILT`

NLI-level compatibility completeness remains partial until Epic B closes it.

## Current Phase

Current order:
1. Epic A done
2. Epic B now
3. Epic C next
4. Graph later

Graph is planned and explicitly not presented as current implementation.

## Setup / Run / Test

```bash
cd /home/ictin_claw/.openclaw/workspace/CognitiveRAG
python3 -m pip install -r CognitiveRAG/requirements.txt
python3 -m uvicorn CognitiveRAG.main_server:app --reload --host 127.0.0.1 --port 8080
python3 -m pytest CognitiveRAG/tests/context_selection -q
python3 -m pytest CognitiveRAG/tests/retrieval -q
python3 -m pytest CognitiveRAG/tests -q
```
