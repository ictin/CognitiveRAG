# CognitiveRAG

**CognitiveRAG is a multi-layer memory and context-construction backend for OpenClaw.**
It helps an agent remember what matters, choose better evidence for the current turn, reuse prior reasoning and promoted knowledge, incorporate local files and web evidence, and explain why specific context was selected.

## What this does for an OpenClaw agent

CognitiveRAG is built to improve the things that break first in ordinary agent systems:

- **Memory continuity**: important conclusions, workflows, and prior reasoning do not disappear when the chat gets long.
- **Better evidence selection**: the system assembles the best context under budget, not just nearest-text or newest-text dumps.
- **Reasoning reuse**: useful prior reasoning and promoted knowledge are brought back when relevant.
- **First-class evidence lanes**: local corpus data, promoted memory, episodic memory, reasoning memory, and web evidence can all contribute in controlled ways.
- **Explainable context construction**: the system can expose what was selected, what was dropped, and why.
- **Bounded discovery**: it can explore useful adjacent evidence without uncontrolled search.

## Why this is different from ordinary RAG

CognitiveRAG is **not** just:
- vector search over chunks,
- chat-memory summarization,
- top-k prompt stuffing,
- or a thin plugin that injects a larger prompt.

It is a backend intelligence layer for OpenClaw that combines multi-layer memory, retrieval lanes, typed candidates, budget-aware selection, promoted knowledge, reasoning reuse, and explainable context assembly.

## Core benefits

- Better turn quality under token constraints.
- Better long-session reliability and recoverability.
- Better reuse of high-value prior knowledge.
- Better evidence grounding across local and web sources.
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
- skill execution/evaluation memory
- web evidence and promoted web memory
- corpus lexical/semantic retrieval lanes

Markdown mirrors are integration artifacts, not the full memory system.

## What is already implemented

- retrieval lanes
- context-selection foundation
- durable promoted memory
- reasoning memory and reuse
- web evidence and promoted web memory
- explanation artifacts
- bounded discovery foundations
- skill-memory execution/evaluation foundations

## What is partial

Epic B parity status:
- B1 typed candidate coverage: `PARTLY_BUILT`
- B2 scoring and token-budget rules: `PARTLY_BUILT`
- B3 contradiction/compatibility filtering: `PARTLY_BUILT`
- B4 reorder and explanation output: `FULLY_BUILT`

NLI-level compatibility completeness remains partial until Epic B parity closes it.

## What is next

Current order:
1. Epic A done
2. Epic B now
3. Epic C next
4. Graph later

Graph is planned and explicitly not presented as current implementation.

## Setup / run / test

```bash
cd /home/ictin_claw/.openclaw/workspace/CognitiveRAG
python3 -m pip install -r CognitiveRAG/requirements.txt
python3 -m uvicorn CognitiveRAG.main_server:app --reload --host 127.0.0.1 --port 8080
python3 -m pytest CognitiveRAG/tests/context_selection -q
python3 -m pytest CognitiveRAG/tests/retrieval -q
python3 -m pytest CognitiveRAG/tests -q
```
