# Phase 3 Acceptance Matrix (Authoritative)

This document defines closure authority for Phase 3:
`permanent and lossless memory system for AI agents`.

It exists to prevent status drift between:
- repo-level correctness,
- runtime-level proof,
- closure-grade decision criteria.

## Closure instrument decision

Decision:
- **Authoritative closure battery:** `tools/run_phase3_closure_battery.py`
- **Heavy benchmark role:** `scripts/run_memory_quality_benchmark.mjs` is **stress telemetry**, not a Phase-3 closure blocker by itself.

Rationale:
- The heavy benchmark includes gateway/session restart lifecycle stress and can fail at boundaries unrelated to Phase-3 memory correctness.
- The closure battery is scoped to Phase-3 acceptance surfaces and produces machine-readable artifacts with explicit checks.

## Acceptance surfaces

Status buckets:
- `DONE_REPO_AND_RUNTIME`
- `DONE_REPO_ONLY`
- `PARTIAL`
- `NOT_DONE`

| Surface | Repo proof required | Runtime proof required | Closure-grade proof required | Current status |
|---|---|---|---|---|
| Conversation store / full session trajectory preservation | session-memory ordering + append tests | live continuity run includes session history round-trip | closure battery continuity checks + session readback | PARTIAL |
| Exact recall and quote/span recovery | selector + recall tests for exact-span preference | live quote-span requests return seeded span | closure battery quote checks, including post-noise requote | DONE_REPO_AND_RUNTIME |
| Task-state continuity | routing/selection tests prefer session-grounded task evidence | live task-state/blockers/next-steps/changed prompts remain session-grounded | closure battery task-state checks + stress continuity | DONE_REPO_AND_RUNTIME |
| Message-part fidelity and tool-trace survival | message-parts store + endpoint tests | runtime readback of tool-trace parts from backend | supplemental runtime readback artifact | PARTIAL |
| Summary lineage / additive compaction | compaction recoverability + deterministic tests | runtime readback where compaction data is surfaced | closure report includes lineage-capable surface | PARTIAL |
| Structured export | structured export tests include parts+compaction fields | runtime export endpoint readback | supplemental runtime readback artifact | PARTIAL |
| Long-session recoverability | compaction/recoverability tests | long continuity runtime pass under stress/noise | closure battery stress continuity + optional telemetry trend | PARTIAL |
| Memory powers context construction (not recall-only) | assemble-context / selector integration tests | live `/crag_explain_memory` + task/quote behavior show memory-to-context use | closure battery explain + continuity checks | DONE_REPO_AND_RUNTIME |

## Required artifacts for Phase 3 closure claim

Required for a top-level Phase-3 closure claim:
1. Closure battery artifact from `tools/run_phase3_closure_battery.py` with all checks passing.
2. Supplemental runtime readback artifact for message-part/tool-trace/export surfaces.
3. Targeted repo regression wall for touched Phase-3 surfaces.
4. Explicit truth-table classification for all Phase-3 surfaces in the package report.

Without all four, top-level Phase-3 closure remains open.

