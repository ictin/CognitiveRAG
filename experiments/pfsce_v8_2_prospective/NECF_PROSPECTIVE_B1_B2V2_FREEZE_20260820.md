# PFSCE v8.2 — NECF Prospective B1 / B2-v2 Freeze

Protocol ID: `PFSCE-V8.2-NECF-PROSPECTIVE-20260820-v1`
Freeze date: 2026-08-20, before the first 10:30 America/New_York prospective issue time.

## Scientific purpose

Test whether the strongest historical PFSCE result survives genuinely unresolved prospective use, and separately test whether the repeatedly positive but non-promoted HRRR weather increment deserves *local-cell* authority. This protocol does not alter any v1 historical verdict.

## Forecast issue time and target

- Issue time: **10:30 America/New_York every day**.
- Eligible BAs: CISO, ERCO, PJM, ISNE.
- Primary continuous target: realized EIA-930 hourly demand for the **12 hourly target intervals immediately following the issue time**, matched to the day-ahead operator forecast available by the issue cutoff.
- Rows whose strong operator baseline is unavailable by the issue cutoff are `BASELINE_UNAVAILABLE` and are retained in the denominator; no substitute is chosen after outcome inspection.
- Target truth is resolved from the first admissible EIA-930 value and later rescored against subsequent vintages for revision sensitivity. First-published and revised scores are never silently merged.

## B1 Anchor A0 — frozen historical policy

Purpose: strongest test of temporal transport without post-2023 model selection.

- Features: exactly the frozen B1 residual/state feature family used in NECF-001: only information at least 48 hours old where lagged state is required, including frozen lag/rolling/calendar construction.
- Ridge alpha grid/model family, HGB specification, nonnegative stack construction, and 2023 router rule remain the historical frozen policy.
- Router action remains: **CISO=PFSCE, PJM=PFSCE, ERCO=operator fallback, ISNE=operator fallback** unless the original frozen executable cannot be reproduced. No 2026 outcomes may alter A0.
- A0 is the anchor for the scientific continuity claim.

## B1 Challenger C1 — simplicity/stability hypothesis

This is a new prospective hypothesis motivated by the repeated 2024/2025 observation that the fixed Ridge residual model outperformed the frozen stack. It receives no retrospective promotion bonus.

- Model: Ridge residual correction only.
- Alpha: **100**, frozen now.
- Feature family: same B1 state/calendar feature set; no HRRR variables.
- Training update rule: expanding point-in-time training using only admissible observations whose truth was available before the current issue time; same 48-hour lag guard; no validation/model-family search after this freeze.
- Eligibility: CISO and PJM only; ERCO and ISNE remain strong-baseline fallback in C1.
- C1 is evaluated against A0 and the operator baseline but cannot rewrite A0 trajectories.

## B2-v2 Challenger W1 — local weather increment

B2-v1 remains a failed family/generalization module. W1 asks a narrower, newly preregistered question: does operational HRRR weather add repeatable incremental skill *within the cells where the historical signal repeated*?

- Eligible cells: **CISO and PJM only**.
- Baseline for incremental scoring: B1 C1 in the same target row.
- Weather source: operational HRRR only; forecast initialization/object must have been available by issue time.
- Weather feature family is frozen to the v1 primitives/derivations: 2 m temperature, 2 m dew point, relative humidity, heat index, and cooling-degree-hours above 18°C using the previously frozen geography adapter.
- Model: Ridge residual correction, alpha **100**, with B1 state/calendar + frozen weather features.
- Training update rule: expanding point-in-time admissible data only. No feature, geography, threshold, or target changes after this freeze.
- ERCO and ISNE are not W1 test cells; they are neither counted as positive nor negative local-cell evidence for W1.

## Local-cell promotion gates

A0 prospective B1 cell can move toward F2 only after a preregistered prospective cohort satisfies all of:

1. at least **30 issue days** and **N_eff >= 20 independent BA-day/episode clusters** for that cell;
2. positive MAE skill versus the operator baseline;
3. BA-day/episode cluster-bootstrap 95% lower bound on relative MAE lift > 0;
4. no material point-in-time/provenance violation;
5. no evidence that the result is dominated by <= 3 issue days;
6. predefined fallback remains intact under missing/OOD inputs.

For W1 B2-v2 local weather promotion in each eligible cell, independently require:

1. at least **30 issue days** and **N_eff >= 20 clusters**;
2. **>= 2%** incremental MAE improvement versus B1 C1;
3. cluster-bootstrap 95% lower bound > 0;
4. positive median issue-day incremental improvement;
5. no >5% degradation in any preregistered heat/non-heat regime slice with >=10 effective clusters unless explicitly marked OOD before scoring;
6. complete HRRR provenance/availability evidence for scored rows.

Cross-cell/generalization authority is a separate claim and is not granted by passing CISO or PJM locally.

## Scoring and trajectory

- Primary continuous metric: MAE; RMSE secondary.
- Report operator, A0, C1, and W1 where eligible.
- Cluster by BA-day and by heat-wave/episode when overlapping days share a latent event; use the more conservative effective-N interpretation.
- Report nominal row count, issue-day count, N_eff, coverage, fallback rate, missing-baseline rate, and OOD rate.
- Preserve every issued forecast and every update append-only. No backfill after truth is known.

## Integrity stop rules

- If live EIA operator forecast cannot be demonstrated available by issue time, record `BASELINE_UNAVAILABLE`; do not use later values.
- If HRRR availability cannot be demonstrated, W1 abstains for that row.
- If an input or code/model generation changes, create a new challenger generation; do not silently merge results.
- Any outcome-aware change to this protocol invalidates confirmatory authority for the affected cohort.
