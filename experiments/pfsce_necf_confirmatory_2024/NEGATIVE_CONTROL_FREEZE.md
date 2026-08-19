# PFSCE / NECF B1 Negative-Control Freeze

Control ID: `NECF-001-B1-NEGATIVE-CONTROLS-v1`

Status: frozen before control scores are inspected. This is a harness/integrity diagnostic using already-resolved 2024 data; it is not additional forecast-skill evidence.

## Purpose

Test whether the B1 pipeline mechanically creates apparent out-of-sample lift when the learnable relationship between forecast-origin features and residual outcomes has been destroyed, and verify that an explicitly future-derived feature is rejected by the temporal guard.

## Controls

### NC1 — permuted residual labels

Use the exact frozen B1 data preparation, quarantine, features, training period (2020-2022), validation period (2023), holdout (2024), model families, alpha grid, HGB specification, stack procedure, router rule and BA-week bootstrap. Before fitting, independently permute residual labels within each balancing authority in TRAIN and VALIDATION using 20 deterministic seeds `2026082001` through `2026082020`. Features and operator baseline are left unchanged. Holdout labels are never permuted.

For every permutation record validation-selected model parameters/router decisions and routed 2024 lift versus operator.

Integrity PASS requires:
- median routed relative MAE lift across permutations <= 2%;
- no more than 2/20 permutations satisfy both routed aggregate lift >=10% and BA-week bootstrap 95% lower bound >0;
- mean lift is materially below the real frozen B1 2024 lift (26.205%).

### NC2 — anti-causal feature rejection

Construct a deliberately invalid candidate feature `future_actual_demand_t_plus_1h` from the target series. Tag its `availability_time` as after forecast origin. The temporal guard must reject the feature before model fitting. A successful predictive score using this feature is a test failure, not a result.

### NC3 — lag-policy invariant

Programmatically enumerate all production B1 state feature names and verify no demand/residual lag shorter than 48 hours exists. Any violation is an integrity failure.

## Interpretation

`NEGATIVE_CONTROLS_PASS` only if NC1, NC2 and NC3 all pass. Failure does not prove the original B1 result is false, but blocks increased authority until the failure is explained and corrected.
