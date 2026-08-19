# PFSCE v8.2 Prospective Validation Program Freeze

Program ID: `PFSCE-V8.2-PROSPECTIVE-20260820`

Freeze time: 2026-08-20 before the first scheduled v8.2 prospective validation cycle.

## Purpose

Establish contamination-resistant prospective evidence for the methodology. This program is required before any module receives durable F2 authority.

## Streams

1. **External benchmark stream** — unresolved ForecastBench questions where accessible under the benchmark's current rules. Preserve benchmark IDs and submit/record forecasts before resolution.
2. **PFSCE general stream** — broad questions generated from a frozen question grammar independently of AOM/economic attractiveness.
3. **NECF dense stream** — repeated load/extreme-load/material-weather questions in fixed cells, plus later attention questions only after material gates permit them.
4. **AOM-directed stream** — economically interesting questions supplied downstream. This stream is scored separately and cannot establish general PFSCE skill by itself.

## Anchor and challenger

- **Anchor**: PFSCE v8.2 methodology, fixed forecasting prompt/protocol family, fixed baseline rules and authority policy. Model/provider/checkpoint must be recorded on every run; a silent provider/model change creates a new anchor generation rather than being merged invisibly.
- **Challenger**: may incorporate later methodology/model improvements, but every change gets a new version and cannot rewrite prior trajectories.

## Required forecast record

Every unresolved forecast stores: question/resolution contract; forecast origin; evidence cutoff and `availability_time`; baseline; raw and calibrated probabilities; probability provenance; authority grade; module passport IDs/MEG; forecastability cell; router decision including baseline fallback/abstention; model/checkpoint/version; evidence lineage/effective N; update triggers; prior trajectory; and exposure/intervention status.

## Scoring

Primary binary metrics: Brier and log loss against the frozen strongest baseline. Report calibration reliability and coverage. Time-to-event targets use censoring-aware proper scoring. Continuous targets use the predeclared proper score/error metric for that family.

Selective-policy reporting must include the full eligible denominator, forecast coverage, fallback coverage, abstention coverage, and skill conditional on each action plus the whole router+forecaster policy.

Trajectory evaluation uses preregistered time-integrated proper-score difference and Early Information Advantage. The weight function and evaluation cadence must be frozen per benchmark family before outcomes.

## Effective sample size

Questions are assigned episode/latent-driver clusters. Report nominal N and an effective independent N estimate. Duplicate questions or multiple horizons for one event are not treated as independent confirmations.

## Promotion minimums

No F2 solely from retrospective evidence. A local module/cell may become F2 only after:

- genuinely unresolved-at-issue prospective forecasts;
- point-in-time evidence integrity;
- frozen strong baseline comparison;
- positive proper-score skill with dependence-aware uncertainty;
- acceptable local calibration for the claimed probability range;
- no material leakage or resolution ambiguity;
- sufficient effective independent N for the claim; and
- predefined OOD/fallback behavior.

Generalization beyond the local cell requires a separately passed V5 transfer test.

## Open-future discovery

At fixed evaluation intervals, independently construct a realized-event set without consulting which questions PFSCE generated. Score Discovery Recall, Discovery Precision, and lead-time-weighted recall. The event-set construction protocol must itself be frozen before matching.

## Stop / integrity rules

- Never backfill a missing forecast after an event becomes obvious.
- Never delete router failures from the denominator.
- Never widen a resolution contract after outcome inspection.
- Any model or methodology update creates a new challenger/anchor generation.
- Prospective results remain pending until their resolution dates; unresolved questions are not scored as failures or successes.
