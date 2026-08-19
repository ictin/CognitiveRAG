# PFSCE v8.2 Cross-Domain Residual Validation — FluSight Freeze

Protocol ID: `PFSCE-V8.2-FLUSIGHT-RESIDUAL-V1`

Status: **FROZEN BEFORE THE 2025-26 HOLDOUT TARGET VALUES OR HOLDOUT SCORES ARE INSPECTED.**

## Purpose

Test whether the v8.2 strong-baseline/residual-forecasting pattern transfers beyond electricity demand into a distinct public-health forecasting domain. This is a V1/V5 diagnostic transfer experiment, not a prospective V3 experiment, because the 2025-26 influenza outcomes already exist at protocol-freeze time.

## Authoritative external source

CDC FluSight Forecast Hub (`cdcepi/FluSight-forecast-hub`). Strong baseline: `FluSight-ensemble` quantile forecasts for target `wk inc flu hosp`. Final evaluation truth: hub `target-data/target-hospital-admissions.csv`.

The source is treated as post-period/revision-sensitive historical evaluation data. No claim of exact real-time target-data vintage authority is made.

## Forecast target and horizons

- Weekly incident confirmed influenza hospital admissions.
- Locations: every location with complete FluSight ensemble forecasts and final truth in the selected seasons.
- Forecast horizons: 1, 2, and 3 weeks ahead only.
- Output: quantile distribution; primary score is mean pinball loss across all common submitted quantiles.
- Secondary scores: median-forecast MAE and central-interval empirical coverage.

## Chronological cohorts

Cohorts are assigned from forecast `reference_date`:

- TRAIN: 2023-10-01 through 2024-06-30.
- VALIDATION / model selection / router: 2024-10-01 through 2025-06-30.
- HOLDOUT: 2025-10-01 through 2026-06-30.

No HOLDOUT outcomes may influence feature design, model selection, hyperparameters, router decisions, promotion thresholds, or interpretation rules.

## Baseline-residual architecture

The baseline is the CDC FluSight ensemble itself. PFSCE does not forecast influenza admissions from zero. It estimates additive residual bias in the ensemble median and shifts the complete baseline quantile distribution by that correction.

For each location-horizon-reference-date case:

`residual = final_truth - baseline_median`

Allowed forecast-origin features are deliberately restricted to information already contained in or mechanically derived from the contemporaneous baseline forecast and calendar identity:

- horizon (categorical),
- location (categorical),
- log1p baseline median,
- reference-date epidemiological/calendar phase encoded as sin/cos annual terms,
- interaction of horizon with log1p baseline median.

No contemporaneous final truth, later revised truth, future hospitalizations, external retrospective covariates, or LLM-generated features are allowed.

## Frozen candidate correction models

1. `NO_CORRECTION`: FluSight ensemble baseline.
2. `HORIZON_BIAS`: mean training residual by horizon, shifted equally across all quantiles.
3. `RIDGE_RESIDUAL`: one-hot location+horizon plus the numeric features above, predicting median residual. Ridge alpha is selected only on VALIDATION from `{0.1, 1, 10, 100, 1000}`.

Model-family selection is performed globally on VALIDATION using mean pinball loss after shifting all quantiles. Ties within 0.25% relative pinball are resolved in favor of the simpler model in the order NO_CORRECTION, HORIZON_BIAS, RIDGE_RESIDUAL.

## Frozen Forecastability Router

After the global challenger family is selected on VALIDATION, routing is by horizon only.

For each horizon 1/2/3, compute weekly-clustered relative pinball improvement of challenger versus baseline on VALIDATION. A horizon is promoted only if the 5,000-resample bootstrap 95% lower bound is strictly greater than zero. Non-promoted horizons use the original FluSight ensemble exactly.

Cluster unit: forecast `reference_date`, preserving dependence across locations and quantiles in the same forecasting round.

## Frozen HOLDOUT promotion gate

The routed policy is a cross-domain **PASS** only if all conditions hold on 2025-26 HOLDOUT:

1. aggregate mean pinball loss improvement versus FluSight ensemble >= 2%;
2. reference-date clustered bootstrap 95% lower bound for aggregate relative pinball improvement > 0;
3. every promoted horizon has non-negative aggregate pinball improvement on HOLDOUT;
4. routed median-forecast MAE does not degrade by more than 1% relative to baseline;
5. mean absolute coverage error across available 50%, 80%, 90%, and 95% central intervals does not worsen by more than 0.02 absolute.

Interpretation:

- `CROSS_DOMAIN_PASS` if all five conditions pass.
- `DIRECTIONAL_TRANSFER` if aggregate proper-score improvement is positive and bootstrap lower bound > 0 but one of conditions 1, 3, 4, or 5 fails.
- `NO_TRANSFER` otherwise.

A PASS supports transfer of the **baseline-residual methodology pattern** to this benchmark family only. It does not validate universal PFSCE, LLM forecasting, DAGs, convergence, social propagation, Mule detection, or open-future discovery.

## Anti-hindsight controls

- This file must be committed before the runner downloads/opens 2025-26 target truth.
- The runner writes source commit SHA, input-file hashes, selected model, alpha, router decisions, and all HOLDOUT metrics.
- Any post-result model redesign becomes `v2` and requires a new untouched/prospective cohort.
- The complete eligible HOLDOUT denominator is retained; failed/missing locations are reported rather than silently deleted.
