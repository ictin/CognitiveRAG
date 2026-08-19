# PFSCE v8.2 Broad Real-World Calibration Validation — ForecastBench

Protocol ID: `PFSCE-V8.2-FORECASTBENCH-MARKET-CAL-V1`

Status: **FROZEN BEFORE THIS PROTOCOL READS OR SCORES THE 2026 HOLDOUT RESOLUTIONS.**

External source: official `forecastingresearch/forecastbench-datasets` plus the public ForecastBench schema in `forecastingresearch/forecastbench`.

Pre-execution schema correction: the public resolution-set files contain resolution fields but intentionally omit the internal `market_value_on_due_date`; therefore this protocol uses the market `freeze_datetime_value` published inside each corresponding LLM question set. ForecastBench freezes questions 10 days before `forecast_due_date`, so this is a genuine earlier market probability and is more temporally conservative than a due-date market quote. This correction was made before any 2026 HOLDOUT resolution rows were materialized or scored by this protocol.

## Purpose

Test a deliberately narrow but important PFSCE claim across heterogeneous real-world questions: whether a time-safe calibration/forecastability layer can add out-of-sample value on top of a strong archived prediction-market probability baseline without harming domains where no stable correction exists.

This is a retrospective historical holdout experiment. It cannot validate full PFSCE reasoning, LLM research, causal DAGs, convergence, open-future discovery, social/narrative modules, or prospective F2 authority.

## Eligible target contract

Read official ForecastBench `datasets/question_sets/*-llm.json` and `datasets/resolution_sets/*_resolution_set.json` files through Git LFS. Pair a resolution set to the LLM question set with the same `forecast_due_date`. For each question set, use only market questions whose published `freeze_datetime_value` is numeric.

Use merged rows satisfying all frozen criteria:

- standard binary target with string question `id`;
- matching `id` and `source` between question and resolution set;
- `resolved == true` where that field is present;
- numeric `resolved_to` exactly 0 or 1;
- numeric question-set `freeze_datetime_value` strictly between 0 and 1;
- parseable question-set `forecast_due_date` and resolution-set `resolution_date`;
- `resolution_date >= forecast_due_date`;
- source is non-missing;
- one unique row per `(id, forecast_due_date, resolution_date, source, direction)` after exact duplicate removal.

Question-set files that cannot be paired to a resolution set are reported and excluded. If the archive contains repeated copies of the same target in multiple files, retain the earliest file instance after sorting by due date and filename. Do not select rows according to model performance or outcome.

The official archived `freeze_datetime_value` is treated as the strong market baseline. ForecastBench defines `FREEZE_DATETIME` ten days before `FORECAST_DATETIME`, so this baseline is intentionally stale by ten days relative to the benchmark forecast deadline. This experiment tests calibration of that frozen market state; it does not claim to beat the market as it stood on the later due date.

## Chronological cohorts

Cohorts are based only on `forecast_due_date`:

- TRAIN: all eligible targets with due date through `2025-06-30`.
- VALIDATION: due date `2025-07-01` through `2025-12-31`.
- HOLDOUT: due date `2026-01-01` through `2026-06-30`.

Targets outside these windows are not scored. No HOLDOUT resolution may affect feature design, regularization, router thresholds, source/horizon bucketing, or PASS criteria.

## Dependence unit

Multiple horizons/directions for one underlying question are correlated. Primary bootstrap cluster = question `id`. Report nominal target rows and unique-question effective N. Combo questions whose IDs are arrays/tuples/lists are excluded by the string-ID rule and do not inflate N.

## Baseline and challengers

All probabilities clipped to `[0.005,0.995]` only for logit/log-loss numerical stability; the same clipping is applied to baseline and challengers.

### M0 — raw frozen-market baseline

`p = freeze_datetime_value` from the published question set.

### M1 — global Platt recalibration diagnostic

Logistic regression of outcome on `logit(p_market)` only.

### M2 — PFSCE metadata calibration challenger

Regularized logistic regression using only information known at question freeze / forecast setup:

- `logit(p_market)`;
- source one-hot;
- `log1p(days_to_resolution)` where days_to_resolution = resolution_date - forecast_due_date;
- horizon bucket one-hot: `<=30d`, `31-90d`, `>90d`;
- absolute market logit (extremity);
- source × market-logit interactions;
- horizon-bucket × market-logit interactions.

No question text, embeddings, modern LLM labels, later market prices, final resolution rationale, or post-freeze outcome information may enter features. Resolution date is part of the frozen question target contract and may be used only to calculate the target horizon.

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

1. routed aggregate Brier improvement vs frozen-market baseline >= **2%**;
2. question-cluster bootstrap 95% lower bound > **0**;
3. routed log loss <= market-baseline log loss;
4. forecast coverage = **100%** including baseline fallbacks;
5. no source with >=100 unique HOLDOUT questions suffers > **5%** Brier degradation under the routed policy.

`BROAD_CALIBRATION_DIRECTIONAL` if aggregate Brier improvement is positive with bootstrap lower bound >0 but another condition fails. Otherwise `BROAD_CALIBRATION_FAIL`.

## Interpretation

A PASS would support only the generality of the PFSCE **calibration + selective routing** pattern over this ForecastBench frozen-market setting. It would not establish that PFSCE beats contemporaneous prediction markets at the forecast deadline because the public baseline is ten days older. A FAIL means no broad correction authority should be granted; local cells may still be researched separately but cannot be promoted from this experiment.

## Integrity rules

- This corrected freeze file must be committed before the workflow materializes/reads 2026 HOLDOUT resolution rows.
- The runner must log the ForecastBench dataset commit SHA, every input question/resolution file hash, row exclusions by reason, paired/unpaired file counts, cohort sizes, selected C values and router decisions.
- Any post-result change to cohort dates, eligibility, features, model family, C grid, cells, bootstrap estimator, thresholds or interpretation requires V2 with a new untouched later/prospective cohort.
