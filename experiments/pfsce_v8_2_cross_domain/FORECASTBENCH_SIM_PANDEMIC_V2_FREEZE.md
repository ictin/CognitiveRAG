# PFSCE v8.2 Controlled-World Validation — ForecastBench-Sim Pandemic v2

Protocol ID: `PFSCE-V8.2-FBSIM-PANDEMIC-V2`

Status: **FROZEN BEFORE ANY V2 SIMULATION OUTCOMES OR V2 HOLDOUT SCORES ARE INSPECTED.**

External benchmark source: `forecastingresearch/forecastbench-sim` pinned to commit `a5b446cdbe6302302bed51616d0ced3b3f5239ed`.

## Why v2 exists

V1 remains a failed frozen experiment and is not reinterpreted. Its router used the arithmetic mean of per-scenario relative Brier ratios. That estimator is unstable when a scenario's baseline Brier is near zero: a tiny denominator can dominate the mean even though the intended estimand is improvement in aggregate proper score. V2 corrects that statistical estimator and uses a completely fresh generated cohort with substantially more independent scenarios.

No V1 holdout observation is reused in V2. V2 does not change the forecasting feature families, strong baseline, challenger family, probability clipping, regularization grid, or PASS thresholds. The scientific changes relative to V1 are explicitly limited to: (a) fresh seed/cohort, (b) larger effective N, and (c) cluster bootstrap of the aggregate score ratio rather than averaging within-cluster ratios.

## Purpose

Replicate the controlled-world test of these PFSCE architectural claims:

1. reference-class baseline;
2. state reconstruction using only the model-facing snapshot;
3. slow/current state versus visible fast trajectory;
4. explicit intervention representation for conditional forecasts;
5. validation-only Forecastability Router with baseline fallback;
6. proper probabilistic scoring and calibration;
7. dependence-aware inference at scenario level.

A favorable result cannot establish real-world F2 authority.

## Fresh generated cohort

Use unchanged benchmark functions `pandemic_world.scenarios.sample_scenarios` and `build_corpus`.

- V2 scenario seed: `20260821`
- total scenarios: `600`
- snapshot day: 20
- horizons: 40 and 60
- four matched questions per scenario: unconditional/conditional × horizon
- expected total questions: 2,400

Frozen split by generated scenario ID:

- TRAIN: IDs `0..299` — 300 scenarios / expected 1,200 questions
- VALIDATION: IDs `300..449` — 150 scenarios / expected 600 questions
- HOLDOUT: IDs `450..599` — 150 scenarios / expected 600 questions

Hidden simulator parameters, scenario seed, future values and resolved values are forbidden as features. Only the benchmark's model-facing context, question kind and horizon are admissible.

## Model ladder — unchanged from v1

All probabilities clipped to `[0.005, 0.995]`.

### B0 reference class
TRAIN empirical rate with Beta(1,1) smoothing by `(kind,horizon)`.

### B1 current-state strong baseline
Logistic regression using only kind, horizon, day-20 cumulative cases, active infections and deaths for both regions, current differences and log ratios. No trajectory slopes or intervention parameters.

### B2 fast-trajectory ablation
B1 plus visible 5-day increments, recent 5/10-day log growth, recent increment difference/ratio, active-to-cumulative fractions and acceleration.

### B3 intervention-aware challenger
B2 plus visible intervention-present flag, vaccine efficacy, coverage, day, efficacy×coverage, timing relative to snapshot, exposure days before target horizon, and efficacy×coverage×exposure interaction. Intervention fields are zero for unconditional questions.

B1/B2/B3 use standardized logistic regression. `C in {0.1,1,10}` is selected independently for each family using VALIDATION mean Brier only. TRAIN is the only fitting cohort. No validation refit.

## Forecastability Router — v2 estimator

Strong baseline: B1. Challenger: B3. Router cells remain `(kind,horizon)`.

For each validation cell, compute observed aggregate relative Brier improvement:

`R = (Brier_B1 - Brier_B3) / Brier_B1`.

Then perform 5,000 **scenario-cluster bootstrap resamples**. Each bootstrap draw samples scenario IDs with replacement; for each sampled scenario include all questions in that cell belonging to that scenario, preserving multiplicity; compute Brier_B1 and Brier_B3 over the entire resampled row set; then compute the aggregate ratio `R*` from those two aggregate scores.

This explicitly does **not** average per-scenario ratios.

Promote B3 in a cell only if:

1. observed aggregate relative Brier improvement > 0; and
2. bootstrap 95% lower bound of `R*` > 0.

Otherwise route to B1 exactly. Full HOLDOUT denominator is retained.

## Primary HOLDOUT scores

- aggregate Brier;
- log loss;
- calibration intercept/slope when identifiable;
- 10-bin ECE descriptive;
- forecast coverage;
- 5,000-resample scenario-cluster bootstrap of aggregate relative Brier improvement using the same v2 estimator.

## Frozen PASS gate — unchanged thresholds

`CONTROLLED_WORLD_PASS` only if all hold on untouched V2 HOLDOUT:

1. routed aggregate Brier improvement versus B1 >= 5%;
2. aggregate scenario-cluster bootstrap 95% lower bound > 0;
3. routed conditional-question Brier improvement versus B1 >= 5%;
4. conditional scenario-cluster bootstrap 95% lower bound > 0;
5. unconditional routed degradation versus B1 no worse than 2%;
6. routed log loss no worse than B1;
7. forecast coverage = 100%.

`CONTROLLED_WORLD_DIRECTIONAL` if aggregate improvement is positive with bootstrap lower bound >0 but at least one other condition fails. Otherwise `CONTROLLED_WORLD_FAIL`.

## Additional frozen diagnostics

Report B0/B1/B2/B3/routed scores and:

- B2 vs B1 aggregate lift;
- B3 vs B2 conditional lift;
- ungated B3 vs B1 aggregate and by-cell lift;
- router vs ungated B3;
- number of promoted cells;
- realized positive-class counts by kind/horizon in TRAIN/VALIDATION/HOLDOUT;
- nominal N and effective independent N = scenario count for each cohort.

## Integrity rules

- This freeze commit must predate V2 simulation execution.
- V1 remains failed and immutable.
- Mechanical-only implementation fixes can be v2.0.x if they do not alter scientific choices.
- Any change to V2 seed, split, features, model families, C grid, router rule, bootstrap estimator, PASS thresholds or interpretation after V2 outcomes are generated requires V3 and another fresh cohort.
