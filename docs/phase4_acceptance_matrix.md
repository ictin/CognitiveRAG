# Phase 4 Acceptance Matrix (Durable Promoted Memory)

This document tracks Phase 4 closure truth for:
`durable promoted memory completion`.

Status buckets:
- `DONE_REPO_AND_RUNTIME`
- `DONE_REPO_ONLY`
- `PARTIAL`
- `NOT_DONE`

## Authoritative Phase 4 validation instrument

- Authoritative runtime battery: `tools/run_phase4_promoted_closure.py`
- Required artifact: `forensics/<stamp>_phase4_promoted_closure/summary.json`

The battery validates:
- promotion trigger path,
- promoted fact/procedure readback,
- provenance/lineage visibility,
- dedup-safe surfacing,
- confidence/freshness surfacing,
- promoted retrieval lane usage,
- truthful episodic-vs-promoted behavior.

## Current surface status (2026-04-12 package)

| Surface | Current status | Evidence |
|---|---|---|
| Stable fact extraction | DONE_REPO_AND_RUNTIME | promotion extractors/normalizers tests + phase4 closure battery |
| Reusable procedure extraction | DONE_REPO_AND_RUNTIME | bridge tests + closure battery procedure checks |
| Speech-act stripping + normalization | DONE_REPO_ONLY | unit tests (`test_speech_act.py`, `test_normalizers.py`) |
| Dedup with provenance retention | DONE_REPO_AND_RUNTIME | dedup tests + promoted readback endpoint/runtime artifact |
| Confidence/freshness fields | DONE_REPO_AND_RUNTIME | normalized/unit + readback + closure battery checks |
| Promoted retrieval in correct mode | DONE_REPO_AND_RUNTIME | policy minimum + promoted lane retrieval + closure battery mode check |
| Local-to-durable lineage traceability | DONE_REPO_AND_RUNTIME | promoted readback endpoints + provenance payloads in runtime artifact |
| Runtime truthfulness (no promoted masquerade) | DONE_REPO_AND_RUNTIME | closure battery episodic guard + promoted search absence checks |

## Remaining closure blockers

Phase 4 remains open at epic level while these remain:
- confidence/freshness policy is currently heuristic and lacks full trust-lifecycle policy integration,
- promotion lifecycle workflows beyond this bridge layer (approval/revalidation/trust workflows) are later-phase scope.
