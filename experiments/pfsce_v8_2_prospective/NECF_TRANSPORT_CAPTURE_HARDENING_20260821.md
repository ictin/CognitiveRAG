# NECF live transport hardening — 2026-08-21

Status: FROZEN TRANSPORT-ONLY CLARIFICATION
Parent protocol: `PFSCE-V8.2-NECF-PROSPECTIVE-20260820-v1`

## Motivation

The 2026-08-20 and 2026-08-21 live attempts demonstrated that a single narrowly timed GitHub/ChatGPT acquisition attempt is operationally fragile. This clarification changes only source-transport redundancy. It does **not** change any forecast target, feature, model, router, score, promotion threshold, or source-eligibility rule.

## Hardening rule

- GitHub Actions may make multiple acquisition attempts between 09:50 and 10:24 America/New_York.
- Every attempt is stored append-only in a unique `attempt-<UTC timestamp>` directory under `necf_precutoff/YYYY-MM-DD/`.
- The frozen forecast cutoff remains 10:30 America/New_York.
- Only source bytes/metadata whose recorded retrieval/availability time is at or before that cutoff are admissible.
- For B1 A0/C1, downstream issue generation may use the **latest pre-cutoff attempt with an eligible EIA capture**.
- For W1, downstream issue generation may independently use the **latest pre-cutoff attempt whose NOAA HRRR F03-F14 availability gate passes**, provided the EIA baseline used for the same forecast was also captured before cutoff.
- Attempts after cutoff, failed attempts, and incomplete attempts remain preserved but cannot be substituted into confirmatory scoring.
- No outcome information may be used to choose among attempts; selection is purely latest-valid-pre-cutoff by the frozen eligibility rules.

## Scientific interpretation

This is reliability engineering around the Time-Capsule boundary, not model tuning. It reduces false loss of prospective observations caused by scheduler delay or runtime networking while preserving the same information set that was legally available before forecast origin.
