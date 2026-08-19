# PFSCE v8.2 Controlled-World Validation — ForecastBench-Sim Pandemic

Protocol ID: `PFSCE-V8.2-FBSIM-PANDEMIC-V1`

Status: **FROZEN BEFORE THIS PROTOCOL'S HOLDOUT SIMULATIONS OR HOLDOUT SCORES ARE INSPECTED.**

External benchmark source: `forecastingresearch/forecastbench-sim` pinned to commit `a5b446cdbe6302302bed51616d0ced3b3f5239ed`.

## Purpose

Test PFSCE architectural claims in an immediately resolvable, contamination-free controlled world, using the ForecastBench-Sim Starsim pandemic world. This experiment specifically evaluates:

1. reference-class baselines;
2. state reconstruction from only the model-facing snapshot report;
3. slow/current state versus fast trajectory features;
4. explicit intervention-aware features for conditional questions;
5. validation-only forecastability routing with baseline fallback;
6. proper probabilistic scoring and calibration diagnostics;
7. dependence-aware uncertainty at the scenario level.

This experiment does **not** validate real-world generality, LLM search/retrieval, narrative propagation, open-future discovery, Mule detection, economic capture, or F2 production authority.

## World and question generation

Use the benchmark's unchanged `pandemic_world.scenarios.sample_scenarios` and `build_corpus` functions.

- scenario seed: `20260820`
- total scenarios: `120`
- snapshot day: benchmark default day `20`
- resolution horizons: benchmark defaults day `40` and day `60`
- each scenario contributes matched unconditional and conditional questions at both horizons
- expected total questions: 480

Chronological-by-generated-ID split, frozen before simulation outcomes are inspected:

- TRAIN scenarios: IDs `0..59` (60 scenarios; expected 240 questions)
- VALIDATION scenarios: IDs `60..89` (30 scenarios; expected 120 questions)
- HOLDOUT scenarios: IDs `90..119` (30 scenarios; expected 120 questions)

The scenario generator's hidden structural parameters and simulation seed may be used only to construct the world. They are **forbidden as forecasting features**. The forecaster may use only fields exposed in the benchmark's model-facing `context` plus question kind and resolution horizon.

## State extraction

Parse only the situation report available to a normal forecaster:

- cumulative cases for Riverton and Southbay at days 0, 5, 10, 15, 20;
- active infections at day 20;
- cumulative deaths at day 20;
- population statement;
- conditional intervention text when present: vaccine day, efficacy, and coverage;
- question horizon (40 or 60) and conditional/unconditional kind.

No future values, resolved values, hidden beta parameters, or simulator internals may enter features.

## Frozen model ladder

All candidate probabilities are clipped to `[0.005, 0.995]` before scoring.

### B0 — Reference class

Empirical TRAIN base rate with Beta(1,1) smoothing, separately by `(kind, horizon)`.

### B1 — Current-state baseline

Logistic regression using only current snapshot state and identity:

- kind;
- horizon;
- log1p cumulative cases R/S at day 20;
- log1p active infections R/S at day 20;
- log1p deaths R/S at day 20;
- current case difference and log case ratio;
- active-infection difference and log active ratio.

No trajectory slopes and no intervention parameter features.

### B2 — Fast-trajectory state

B1 plus features mechanically derived from the visible case trajectory:

- 5-day increments for intervals 0-5, 5-10, 10-15, 15-20 for both regions;
- recent 5-day and 10-day log growth for both regions;
- recent increment difference and ratio;
- active/cumulative-case fractions;
- acceleration from prior to recent 5-day increment.

### B3 — Intervention-aware state

B2 plus visible intervention features:

- intervention-present flag;
- vaccine efficacy;
- vaccine coverage;
- vaccine day;
- efficacy × coverage;
- days from snapshot to vaccine day;
- vaccine-already-started flag;
- days of intervention exposure before the question's resolution horizon;
- efficacy × coverage × exposure-days interaction.

For unconditional questions these intervention features are zeroed because no intervention is presented in the model-facing context.

## Fitting and regularization

B1/B2/B3 use scikit-learn logistic regression with standardized numeric features and one-hot categorical identity where needed.

Regularization candidates: `C in {0.1, 1.0, 10.0}`. For each model family, choose C using VALIDATION mean Brier only. TRAIN is the only fitting set. The selected C is then frozen for HOLDOUT; no refit on validation.

## Strong baseline and forecastability router

The strong baseline is B1 current-state.

The PFSCE challenger is B3 intervention-aware state. B2 is an ablation diagnostic and may not replace B3 after holdout inspection.

Router cells are `(kind, horizon)`, four cells total. A cell is promoted from B1 to B3 only when, on VALIDATION:

1. mean Brier improvement of B3 over B1 is positive; and
2. the 5,000-resample scenario-cluster bootstrap 95% lower bound of relative Brier improvement is strictly greater than zero.

All other cells fall back to B1 exactly. The entire eligible HOLDOUT denominator remains in the routed-policy score.

## Frozen primary scores

- mean Brier score;
- log loss;
- calibration intercept/slope where identifiable;
- 10-bin expected calibration error (ECE) as descriptive;
- forecast coverage (must be 100%; no missing forecasts allowed);
- scenario-cluster bootstrap over the 30 HOLDOUT scenarios.

Questions from the same scenario are one dependence cluster.

## Frozen PASS gate

`CONTROLLED_WORLD_PASS` only if all hold on untouched HOLDOUT:

1. routed aggregate Brier improvement versus B1 >= **5%**;
2. scenario-cluster bootstrap 95% lower bound for routed relative Brier improvement > **0**;
3. conditional-question routed Brier improvement versus B1 >= **5%**;
4. conditional-question scenario-cluster bootstrap 95% lower bound > **0**;
5. unconditional routed Brier degradation versus B1 is no worse than **2%**;
6. routed log loss is no worse than B1;
7. forecast coverage = **100%**.

`CONTROLLED_WORLD_DIRECTIONAL` if aggregate Brier improvement is positive with bootstrap lower bound >0 but at least one other gate condition fails.

Otherwise: `CONTROLLED_WORLD_FAIL`.

## Ablation interpretation

Record B0, B1, B2, B3 and routed-policy scores on HOLDOUT. The following are descriptive mechanism diagnostics, not separate promotion gates:

- B2 minus B1 estimates value of fast trajectory features;
- B3 minus B2 on conditional questions estimates value of explicit intervention representation;
- routed policy minus ungated B3 estimates the safety value of the Forecastability Router.

## Integrity rules

- The freeze file commit must predate execution of this protocol's holdout simulations and scoring.
- Any bug fix that changes only data type/serialization/control flow may be versioned `v1.0.x` and must document that no scientific parameter changed.
- Any feature, split, threshold, model-family, regularization grid, router, or interpretation change becomes `v2` and requires a fresh untouched generated cohort with a different seed.
- A favorable controlled-world result cannot produce F2 real-world authority; it only validates the relevant architecture under a controlled simulator.
