# PFSCE v8.2 NECF First-Run Integrity Audit Freeze

Audit ID: `PFSCE-V8.2-NECF-FIRST-RUN-AUDIT-20260820`
Freeze time: 2026-08-20T08:53Z, before the first 10:30 America/New_York issue time.

This audit evaluates process integrity only. It must not use first-day predictive performance to modify model rules, gates, target windows, router actions, features, or training policy.

## Required checks

1. **Issue-time integrity** — forecast packet timestamp is at or before the frozen 10:30 America/New_York issue cutoff and before target truth availability.
2. **Target-window integrity** — exactly the next 12 eligible hourly target intervals are recorded using the frozen time-zone convention, with DST/UTC conversion explicit.
3. **Operator-baseline provenance** — every baseline value has an availability timestamp/source proving it was available by issue cutoff. Later operator values are prohibited as issue-time baselines.
4. **Denominator preservation** — `BASELINE_UNAVAILABLE`, missing-feature, abstention, OOD, and fallback rows remain in the eligible denominator.
5. **A0 immutability** — CISO/PJM PFSCE route and ERCO/ISNE operator fallback are unchanged; frozen model/specification identifiers match the preregistration.
6. **C1 immutability** — Ridge-only alpha=100, same frozen B1 state/calendar feature family, CISO/PJM eligibility only; no first-day tuning.
7. **W1 HRRR provenance** — HRRR init/object availability precedes issue time; frozen geography and TMP/DPT/RH/heat-index/CDH feature definitions are used; no later/reanalysis weather enters.
8. **48-hour state guard** — every state/residual lag or rolling statistic that requires the historical guard excludes information newer than allowed by the frozen design.
9. **Model-generation provenance** — executable/code SHA, model-generation ID, training cutoff, data snapshot/version and environment/dependency manifest are recorded.
10. **Truth separation** — no target truth is accessed during forecast generation. Resolution occurs in a later stage and stores first-published truth separately from revisions.
11. **Append-only ledger** — issued A0/C1/W1 predictions are immutable. Corrections after issue are new records with reason codes, never overwrites.
12. **Router/fallback behavior** — every target row records route, fallback, abstention and OOD reason; silent dropping is a hard failure.
13. **Prediction sanity** — finite values, no impossible timestamp duplication, no missing BA/target key, and no obvious unit mismatch. Sanity failures trigger quarantine, not model retuning.
14. **Leak audit** — search all materialized inputs for timestamps/publication times after forecast origin and for target-like fields accidentally merged into features.
15. **Reproducibility** — a clean rerun from the frozen issue-time snapshot reproduces issued predictions within declared numerical tolerance.

## Verdict

- `INTEGRITY_PASS`: all hard checks pass; minor documented non-authority metadata omissions may be repaired append-only.
- `INTEGRITY_PASS_WITH_QUARANTINE`: pipeline is valid but explicitly identified rows are quarantined for frozen, non-performance-related reasons.
- `INTEGRITY_FAIL`: any future data, post-cutoff baseline/weather, denominator deletion, silent overwrite, target leakage, model/gate change, or non-reproducible issued prediction.

An `INTEGRITY_FAIL` invalidates the affected prospective issue day for confirmatory authority but the failed day remains in the audit history. Repairs create a new generation and cannot retroactively convert invalid forecasts into valid first-issue forecasts.
