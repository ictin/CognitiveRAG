# CognitiveRAG

**CognitiveRAG is a multi-layer memory and context-construction backend for OpenClaw.**
It helps an agent remember what matters, choose better evidence for the current turn, reuse prior reasoning and promoted knowledge, incorporate local files and web evidence, and explain why specific context was selected.

## Why this matters

Most agent systems fail in the same places:
- they forget important earlier work
- they stuff the prompt with too much weak context
- they fail to reuse good prior reasoning
- they blur together temporary evidence and durable knowledge
- they cannot clearly explain why a given piece of context was included

CognitiveRAG is built to fix those problems.

## What this does for an OpenClaw agent

CognitiveRAG improves the things that matter in real use:

- **Memory continuity**: important conclusions, workflows, and prior reasoning stay reusable instead of disappearing into long chats.
- **Better evidence selection**: the system tries to assemble the best context under budget, not just the nearest or newest text.
- **Reasoning reuse**: useful prior reasoning and promoted knowledge can be brought back when relevant.
- **First-class evidence lanes**: local corpus data, promoted memory, episodic memory, reasoning memory, and web evidence can all contribute in controlled ways.
- **Explainable context construction**: the system can expose what was selected, what was dropped, and why.
- **Bounded discovery**: it can explore adjacent evidence without turning into uncontrolled search.

## Why this is different from ordinary RAG

CognitiveRAG is not just:
- vector search over chunks
- chat-memory summarization
- top-k prompt stuffing
- or a bigger prompt wrapped around a chat interface

It is a backend intelligence layer for OpenClaw that combines multi-layer memory, retrieval lanes, typed candidates, budget-aware selection, promoted knowledge, reasoning reuse, and explainable context assembly.

## Core benefits

- Better turn quality under token limits.
- Better long-session reliability and recoverability.
- Better reuse of high-value prior knowledge.
- Better grounding across local and web evidence.
- Better auditability of context decisions.

## Current architecture

Primary backend surfaces:
- API and routes: `CognitiveRAG/main_server.py`, `CognitiveRAG/api/routes/*`
- Retrieval lanes and routing: `CognitiveRAG/crag/retrieval/*`
- Context selection and explanation outputs: `CognitiveRAG/crag/context_selection/*`
- Session memory compaction/recovery: `CognitiveRAG/session_memory/*`
- Promoted/reasoning/web/skill memory layers: `CognitiveRAG/memory/*`, `CognitiveRAG/web_memory/*`, `CognitiveRAG/skill_memory/*`

## Memory taxonomy

Current layers include:
- session and episodic memory
- promoted memory
- reasoning memory
- skill execution and evaluation memory
- web evidence and promoted web memory
- corpus lexical and semantic retrieval lanes

Markdown mirrors are integration artifacts, not the full memory system.

## What is already real

- retrieval lanes
- context-selection foundation
- promoted memory durability
- reasoning memory and reuse
- web evidence and promoted web memory
- explanation artifacts
- bounded discovery foundations
- skill-memory execution/evaluation foundations

## What is partial

Epic B parity status:
- B1 typed candidate coverage: `FULLY_BUILT`
- B2 scoring and token-budget rules: `FULLY_BUILT`
- B3 contradiction/compatibility filtering: `PARTLY_BUILT`
- B4 reorder and explanation output: `FULLY_BUILT`

NLI-level compatibility completeness remains partial until Epic B parity closes it.
Current B3 behavior includes deterministic contradiction threshold drops and heuristic pairwise compatibility gating; NLI-backed compatibility is not implemented yet.
Current active Epic B step is B3 (contradiction and compatibility filtering).

## What is later

- graph layer and graph-assisted retrieval
- deeper trust/promotion lifecycle expansion
- federation/hive-style extensions

Graph is planned, not current implementation.

## Setup / run / test

```bash
cd /home/ictin_claw/.openclaw/workspace/CognitiveRAG
python3 -m pip install -r CognitiveRAG/requirements.txt
python3 -m uvicorn CognitiveRAG.main_server:app --reload --host 127.0.0.1 --port 8080
python3 -m pytest CognitiveRAG/tests/context_selection -q
python3 -m pytest CognitiveRAG/tests/retrieval -q
python3 -m pytest CognitiveRAG/tests -q
```
