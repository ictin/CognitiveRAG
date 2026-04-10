# CognitiveRAG

**CognitiveRAG is a multi-layer memory and context-construction backend for OpenClaw.**  
It helps an agent remember what matters, choose better evidence for the current turn, reuse prior reasoning and promoted knowledge, incorporate local files and web evidence, and explain why specific context was selected.

\---

## Why this matters

Most agent systems break in the same places:

* they forget important earlier work
* they stuff the prompt with too much weak context
* they fail to reuse good prior reasoning
* they blur together temporary evidence and durable knowledge
* they cannot clearly explain why a given piece of context was included

CognitiveRAG exists to fix those problems.

Instead of treating memory as a single blob or treating retrieval as simple top-k chunk search, CognitiveRAG tries to assemble the **best context for the current turn** under real token and runtime limits.

\---

## What this does for an OpenClaw agent

CognitiveRAG improves the parts of agent behavior that matter in real use:

* **Memory continuity**  
Important conclusions, workflows, and prior reasoning stay reusable instead of disappearing into long chats.
* **Better evidence selection**  
The system tries to assemble the best context under budget, not just the newest or nearest text.
* **Reasoning reuse**  
Useful prior reasoning and promoted knowledge can be brought back when they are relevant again.
* **First-class evidence lanes**  
Local corpus data, promoted memory, episodic memory, reasoning memory, and web evidence can all contribute in controlled ways.
* **Explainable context construction**  
The system can expose what was selected, what was dropped, and why.
* **Bounded discovery**  
It can explore adjacent evidence when useful without turning into uncontrolled search.

\---

## Why this is different from ordinary RAG

CognitiveRAG is **not** just:

* vector search over chunks
* chat-memory summarization
* top-k prompt stuffing
* a plugin that injects a larger prompt
* a thin wrapper around file search

It is a backend intelligence layer for OpenClaw that combines:

* multi-layer memory
* retrieval lanes
* typed candidates
* budget-aware selection
* promoted knowledge
* reasoning reuse
* web evidence memory
* bounded discovery
* explanation artifacts

The goal is not to return the most text.  
The goal is to return the **most useful context**.

\---

## Core benefits

### 1\. Better memory without fake “perfect recall”

CognitiveRAG is designed so important work can survive beyond the current prompt window. That includes conclusions, procedures, reasoning traces, promoted knowledge, and useful evidence.

### 2\. Better evidence under budget

Long context windows are expensive and noisy. CognitiveRAG tries to make context selection deliberate instead of dumping everything into the model.

### 3\. Better reuse of what already worked

If the agent already solved something well, the system should help it reuse that work instead of rediscovering it from scratch.

### 4\. Better use of multiple source types

A good agent should be able to combine conversation memory, durable knowledge, local documents, and web evidence without pretending they are all the same thing.

### 5\. Better explainability

When context was selected, the system should be able to say what was selected, what was dropped, and why.

\---

## Current architecture

At a high level, CognitiveRAG has five main jobs:

1. **Store and retrieve important memory**
2. **Build typed candidates across multiple evidence lanes**
3. **Score and select context under policy and token budgets**
4. **Promote durable knowledge and reasoning for later reuse**
5. **Support bounded discovery and explanation**

A simplified flow looks like this:

1. receive a user request
2. interpret the likely retrieval intent
3. gather candidates from relevant memory and evidence lanes
4. normalize them into a shared candidate shape
5. prune weak or redundant items
6. score them under intent and policy
7. select the best budget-aware context
8. reorder context for stronger model use
9. emit explanation artifacts alongside the selected context

\---

## Memory taxonomy

CognitiveRAG is built around multiple memory and evidence classes instead of one flat store.

### Working memory

Transient active context for the current turn.

### Episodic memory

Conversation history, turn sequence, and recoverable prior interaction state.

### Semantic memory

Durable normalized facts and stable project knowledge.

### Procedural memory

Reusable know-how, workflows, and operational patterns.

### Task memory

Current objective, blockers, subtasks, and next steps.

### Profile memory

Stable user and agent preferences, assumptions, and constraints.

### Reasoning memory

Reusable cognitive work such as reasoning traces, evidence bundles, and successful prior solutions.

### Corpus memory

Local books, docs, and chunked indexed files used as first-class evidence.

### Large-file memory

Oversized local sources stored in a way that preserves exact spans without stuffing prompts.

### Web evidence memory

Fresh external evidence captured for current use.

### Web promoted memory

External knowledge that has been explicitly promoted for durable reuse.

\---

## What is already real

The current implemented base is already substantial.

### Built enough to rely on

* retrieval lanes
* context-selection foundation
* durable promoted memory
* reasoning memory and reuse
* web evidence memory
* promoted web memory
* bounded discovery foundations
* explanation artifacts
* skill-memory related pieces
* live plugin integration path
* live acceptance harness and signoff flow

### What that means in practice

Today, the system can already:

* retrieve across multiple memory classes
* keep durable promoted knowledge
* promote and reuse reasoning
* distinguish staged web evidence from promoted web knowledge
* assemble context with explicit selection behavior
* surface selection/explanation artifacts
* operate through the real OpenClaw integration path

\---

## What is partial

This project is real, but not “finished.”

The main known partial areas are:

* full typed candidate parity across all intended lanes
* full parity for global constrained selection vs the canonical ideal
* contradiction / compatibility handling beyond the current implemented approach
* some parity gaps between intended design and exact code path details
* broader trust, lifecycle, and promotion workflows

These are not hidden. They are current work.

\---

## What is not started yet

Some important ideas are later-phase work, not current claims:

* graph layer implementation
* graph-assisted retrieval
* broader trust and revalidation workflows
* federation / hive-mind style knowledge sharing
* larger graph-based clustering and category/topic systems

This repo should not present those as already built.

\---

## Current status

Live runtime signoff is complete.  
The current project focus is **design/code parity**: making sure the real implementation matches the intended context-selection and retrieval design cleanly and truthfully.

In simple terms:

* the runtime path is proven
* the system is usable
* the next job is to tighten parity and close gaps, not to restart from zero

\---

## Repository role in the OpenClaw ecosystem

This repository is the **backend intelligence layer**.

It owns:

* memory logic
* retrieval logic
* ranking and selection
* promoted memory
* reasoning reuse
* discovery logic
* explanation artifacts
* later graph and trust logic

It does **not** own:

* OpenClaw plugin registration
* OpenClaw command wiring
* smoke/live adapter scripts
* runtime proof in the plugin layer

Those belong in the adapter repository.

\---

## Project layout

Typical important areas include:

* `crag/` — core backend logic
* `crag/context\_selection/` — candidate building, pruning, scoring, selection, reorder, explanations
* `crag/retrieval/` — retrieval routing and lane integration
* `crag/contracts/` — shared schemas and enums
* `crag/cognition/` — contradiction and related cognitive helpers
* `session\_memory/` — context assembly and session-related memory flows
* `tests/` — backend tests for retrieval, selector behavior, contracts, and parity

\---

## Setup

Clone the repository:

```bash
git clone https://github.com/ictin/CognitiveRAG.git
cd CognitiveRAG
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and test dependencies using the repo’s current packaging setup.

If your local environment uses a nested package root, keep the package path explicit when needed.

\---

## Run tests

A typical targeted test flow looks like this:

```bash
pytest -q
```

For targeted parity work, run only the relevant selector, retrieval, or contract tests first, then broaden out.

When changing runtime-sensitive behavior, repo tests are necessary but not enough. The matching OpenClaw integration layer must also prove the live path.

\---

## Development principles

This repository follows a few core rules:

* backend owns intelligence
* selection should favor the best tokens, not the most tokens
* provenance should survive transformations
* discovery must stay bounded
* mirrors are support surfaces, not the whole memory system
* no repo-only success for runtime behavior

\---

## Relationship to the plugin repo

CognitiveRAG is the backend.  
The OpenClaw adapter lives here:

* `https://github.com/ictin/openclaw-cognitiverag-memory`

That adapter owns:

* OpenClaw-facing integration
* context-engine registration
* runtime-proof and live acceptance surfaces
* fail-open behavior in the plugin layer

This backend owns the actual memory and context intelligence.

\---

## Roadmap direction

The current direction is:

1. finish parity between intended and implemented context construction
2. add stronger metrics, smoke, and regression safety
3. only then move into the first graph layer
4. later expand into trust workflows, lifecycle controls, and broader graph-assisted behavior

\---

## Why this project matters

If you want an OpenClaw agent that:

* remembers important work,
* selects better evidence,
* reuses prior reasoning,
* uses both local and external evidence,
* and can explain its own context decisions,

then you need more than ordinary RAG and more than ordinary chat memory.

That is what CognitiveRAG is trying to provide.

