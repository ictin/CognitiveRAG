# 2025 Replication Contract — Frozen Before 2025 Outcome Scoring

Protocol family: `NECF-001-WP2-CONFIRMATORY-2024-v1`  
Replication ID: `NECF-001-WP2-REPLICATION-2025-v1`

## Purpose

Replicate the already-frozen B1 and B2 systems on a completely later calendar cohort without using 2024 or 2025 outcomes for training, hyperparameter selection, stack selection, router selection, geography selection, feature selection, or gate modification.

The 2024 B1 result is already frozen as strong provisional evidence. The 2024 B2 result is already frozen as **FAIL** under its preregistered promotion gate despite a positive aggregate incremental lift. This replication cannot retroactively change either 2024 verdict.

## Model freeze

### B1

- Training cohort remains 2020-01-01 through 2022-12-31 UTC.
- Validation/router cohort remains 2023-01-01 through 2023-12-31 UTC.
- All B1 feature definitions, 48/72/168-hour lags, rolling windows, ridge alpha grid, HGB hyperparameters, non-negative MAE stack, 10:30 America/New_York origin, next-12-hour target window, quarantine rules, and router rule remain unchanged.
- The implementation must reproduce the already-frozen B1 validation decisions before 2025 scoring: CISO and PJM promoted; ERCO and ISNE fall back.
- 2024 data must not be used for fitting or model selection.

### B2

- B2 is the frozen B1 system plus exactly the five already-frozen HRRR-derived weather features: temperature, dew point, relative humidity, heat index, and cooling-degree-hours over 18 C.
- Training and validation remain 2020-2022 / 2023 exactly as in B2 2024.
- Geography remains four equal-weight load centres per BA.
- HRRR access remains `POINT_IN_TIME_ELIGIBLE_DERIVED_HRRR_MIRROR`: original NOAA `noaa-hrrr-bdp-pds` object timestamps govern eligibility; numeric TMP/DPT values come from the frozen `hrrrzarr` representation.
- The implementation must reproduce the frozen B2 validation router decisions before 2025 scoring: CISO and PJM promoted; ERCO and ISNE fall back.
- 2024 data must not be used for fitting or model selection.

## 2025 EIA replication source

The replication target will use a separately hash-verified fixed PUDL/Zenodo EIA-930 archive published after the complete 2025 calendar year:

- Zenodo record: `18448416`, published 2026-02-01, PUDL EIA-930 archive version 62.0.0.
- `eia930-2025half1.zip` MD5: `4ebc17b3e786b5c4a06315afc285093f`
- `eia930-2025half2.zip` MD5: `9a7fe2bb3cdfe42236782ab7291a49ba`

Only 2025 rows from this replication archive may be appended to the original fixed 2020-2024 source used for B1/B2 fitting. Historical 2020-2023 training/validation rows must continue to come from the original frozen record `14881638`; they must not silently be replaced by revised values from the 2026 archive.

The 2025 source is classified `V1_PROVISIONAL_REVISION_SENSITIVE_REPLICATION` because it is a post-period archive and historical EIA-930 raw files are known to be revision-sensitive.

## 2025 weather source

Extract 2025 operational HRRR using the already-frozen B2 source adapter. Include 2024-12-31 weather only to cover the UTC boundary implied by the already-frozen UTC cohort split. No 2024 demand outcome is admitted to fitting.

## B1 replication interpretation gate

This gate is frozen before 2025 outcomes are opened. It does not alter the original 2024 B1 verdict.

A **strong B1 replication** requires all of:

1. routed 2025 MAE improvement over the EIA operator forecast >= 10%;
2. BA-week bootstrap 95% lower bound for routed relative lift > 0;
3. both validation-promoted BAs, CISO and PJM, have positive aggregate MAE improvement over the operator forecast;
4. fallback BAs are not altered by the routed system.

If conditions 2-4 hold but aggregate improvement is positive and <10%, label the result **directional replication**, not strong replication. Otherwise label it **replication failure**.

## B2 replication interpretation

The exact frozen 2024 B2 incremental gate is re-applied descriptively to 2025:

1. aggregate routed MAE improvement over frozen B1 >= 2%;
2. paired BA-week bootstrap 95% lower bound > 0;
3. at least 3 of 4 BAs have positive median weekly improvement;
4. maximum degradation among B2-unrouted BAs <= 5%.

Because B2 failed its 2024 confirmatory gate, even a 2025 pass cannot retroactively promote B2. A 2025 pass would justify a separately preregistered B2-v2 confirmatory experiment; another fail strengthens the conclusion that the present B2 routing/geography/weather representation is not sufficiently broad or stable for promotion.

## Stop rule

Freeze and report the 2025 B1 and B2 replication results before any model redesign. Do not proceed to narrative/social layers on the authority of the current B2-v1 system.