# EIA-930 Revision Sensitivity Audit Freeze

Audit ID: `NECF-001-EIA930-REVISION-AUDIT-v1`

Frozen before row-level vintage comparison results are inspected.

## Purpose

Quantify whether post-publication revisions to EIA-930 historical archives materially affect the 2024 B1 confirmatory conclusion. This audit does **not** convert revised history into point-in-time evidence; it measures robustness of the already provisional result.

## Row-level 2024 vintage panel

Compare exact 2024 demand and day-ahead demand-forecast values for CISO, ERCO, PJM and ISNE across these immutable Zenodo records:

- `14881638` — original source used by the frozen 2024 confirmatory run
- `14949257`
- `15568995`
- `15877505`
- `17241309`
- `17846570`
- `18448416`
- `19367770`
- `19787334`

For each vintage and each BA report: common rows, missing/extra rows, exact changed demand rows, exact changed forecast rows, changed-either rows, absolute delta distribution, and maximum deltas.

## Full-pipeline sensitivity panel

Re-run the frozen B1 architecture independently using each of these four spanning vintages:

- `14881638`
- `17241309`
- `18448416`
- `19787334`

For every vintage:

- train: 2020–2022
- validation/model selection/router: 2023 only
- holdout: 2024 only
- same quarantine rule
- same lag/rolling feature policy
- same ridge alpha candidate grid
- same HGB specification
- same nonnegative stack procedure
- same BA-week router rule
- same 5,000-resample BA-week bootstrap
- no 2025 data may enter

This panel is a **revision-sensitivity diagnostic**, not a new confirmatory experiment, because later vintages were selected after the 2024 outcome existed.

## Frozen interpretation

`ROBUST_TO_REVISION` iff every full-pipeline vintage satisfies all of:

1. routed aggregate MAE improvement versus its own operator baseline >= 10%;
2. BA-week bootstrap 95% lower bound > 0;
3. no vintage reverses the aggregate sign.

`FRAGILE_TO_REVISION` if any full-pipeline vintage has routed aggregate MAE improvement <= 0 or BA-week bootstrap 95% lower bound <= 0.

Otherwise: `REVISION_SENSITIVE_BUT_DIRECTIONALLY_ROBUST`.

Separately record the spread in routed relative MAE lift across vintages and whether validation router decisions change. Even `ROBUST_TO_REVISION` remains `V1_PROVISIONAL_REVISION_SENSITIVE`; it cannot establish point-in-time historical authority.
