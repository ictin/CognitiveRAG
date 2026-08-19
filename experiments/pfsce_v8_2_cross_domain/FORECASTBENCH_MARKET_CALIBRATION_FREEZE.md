# PFSCE v8.2 Broad Real-World Calibration Validation — ForecastBench

Protocol ID: `PFSCE-V8.2-FORECASTBENCH-MARKET-CAL-V1`

Status: **FROZEN BEFORE THIS PROTOCOL READS OR SCORES THE 2026 HOLDOUT RESOLUTIONS.**

External source: official `forecastingresearch/forecastbench-datasets` plus the public ForecastBench schema in `forecastingresearch/forecastbench`.

## Purpose

Test a deliberately narrow but important PFSCE claim across heterogeneous real-world questions: whether a time-safe calibration/forecastability layer can add out-of-sample value on top of an already-strong prediction-market probability baseline without harming domains where no stable correction exists.

This is a retrospective historical holdout experiment. It cannot validate full PFSCE reasoning, LLM research, causal DAGs, convergence, open-future discovery, social/narrative modules, or prospective F2 authority.

## Eligible target contract

Read official ForecastBench `datasets/resolution_sets/*_resolution_set.json` files through Git LFS. Use rows satisfying all frozen criteria:

- standard binary target with string question `id`;
- `resolved == true` where that field is present;
- numeric `resolved_to` exactly 0 or 1;
- numeric `market_value_on_due_date` strictly between 0 and 1;
- parseable `forecast_due_date` and `resolution_date`;
- `resolution_date >= forecast_due_date`;
- source is non-missing;
- one unique row per `(id, forecast_due_date, resolution_date, source, direction)` after exact duplicate removal.

If the archive contains repeated copies of the same target in multiple files, retain the earliest file instance after sorting by resolution-set filename. Do not select rows according to model performance or outcome.

The official archived `market_value_on_due_date` is treated as the strongest baseline available in this benchmark. This experiment does not independently reconstruct the underlying market's tick-level point-in-time history.

## Chronological cohorts

Cohorts are based only on `forecast_due_date`:

- TRAIN: all eligible targets with due date through `2025-06-30`.
- VALIDATION: due date `2025-07-01` through `2025-12-31`.
- HOLDOUT: due date `2026-01-01` through `2026-06-30`.

Targets outside these windows are not scored. No HOLDOUT resolution may affect feature design, regularization, router thresholds, source/horizon bucketing, or PASS criteria.

## Dependence unit

Multiple horizons/directions for one underlying question are correlated. Primary bootstrap cluster = question `id`. Report nominal target rows and unique-question effective N. Repeated/combo targets excluded by the string-ID rule do not inflate N.

## Baseline and challengers

All probabilities clipped to `[0.005,0.995]` only for logit/log-loss numerical stability; the same clipping is applied to baseline and challengers.

### M0 — raw market baseline

`p = market_value_on_due_date`.

### M1 — global Platt recalibration diagnostic

Logistic regression of outcome on `logit(p_market)` only.

### M2 — PFSCE metadata calibration challenger

Regularized logistic regression using only information available by the forecast due date:

- `logit(p_market)`;
- source one-hot;
- `log1p(days_to_resolution)` where days_to_resolution = resolution_date - forecast_due_date;
- horizon bucket one-hot: `<=30d`, `31-90d`, `>90d`;
- absolute market logit (extremity);
- source × market-logit interactions;
- horizon-bucket × market-logit interactions.

No question text, embeddings, modern LLM labels, later market prices, final resolution rationale, or post-due-date information may enter features.

For M1 and M2 independently choose logistic `C` from `{0.01,0.1,1,10}` using VALIDATION aggregate Brier only. TRAIN is the only fitting cohort. Do not refit on VALIDATION.

## Forecastability Router

Strong baseline = M0. Challenger = M2.

Frozen router cell: `(source, horizon_bucket)`. A cell is promoted only when VALIDATION satisfies all:

1. at least 50 unique question IDs;
2. at least 5 positive and 5 negative outcomes;
3. observed aggregate Brier improvement of M2 versus M0 > 0;
4. 5,000-resample question-cluster bootstrap 95% lower bound of aggregate relative Brier improvement > 0.

Cluster bootstrap resamples question IDs with replacement and recomputes aggregate baseline and challenger Brier across all rows belonging to sampled IDs. It does not average per-question relative ratios.

All non-promoted cells use M0 exactly. The complete eligible HOLDOUT denominator is retained; unseen sources/cells automatically fall back to M0.

## Primary HOLDOUT metrics

- aggregate Brier and log loss for M0/M1/M2/routed policy;
- relative Brier improvement of routed policy vs M0;
- 5,000-resample question-cluster bootstrap CI for aggregate relative improvement;
- calibration intercept/slope where identifiable;
- 10-bin ECE descriptive;
- coverage/fallback/promotion shares;
- source/horizon-cell results;
- nominal N and unique-question effective N.

## Frozen PASS gate

`BROAD_CALIBRATION_PASS` only if all hold on untouched HOLDOUT:

1. routed aggregate Brier improvement vs market >= **2%**;
2. question-cluster bootstrap 95% lower bound > **0**;
3. routed log loss <= market log loss;
4. forecast coverage = **100%** including baseline fallbacks;
5. no source with >=100 unique HOLDOUT questions suffers > **5%** Brier degradation under the routed policy.

`BROAD_CALIBRATION_DIRECTIONAL` if aggregate Brier improvement is positive with bootstrap lower bound >0 but another condition fails. Otherwise `BROAD_CALIBRATION_FAIL`.

## Interpretation

A PASS would support only the generality of the PFSCE **calibration + selective routing** pattern over this ForecastBench market-baseline setting. A FAIL means no broad correction authority should be granted; local cells may still be researched separately but cannot be promoted from this experiment.

## Integrity rules

- This freeze file must be committed before the workflow materializes/reads 2026 HOLDOUT resolution rows.
- The runner must log the ForecastBench dataset commit SHA, every input resolution-set file hash, row exclusions by reason, cohort sizes, selected C values and router decisions.
- Any post-result change to cohort dates, eligibility, features, model family, C grid, cells, bootstrap estimator, thresholds or interpretation requires V2 with a new untouched later/prospective cohort.
