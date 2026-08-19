# B1 2024 Confirmatory Verdict — NECF-001-WP2-CONFIRMATORY-2024-v1

**Status:** FROZEN AFTER 2024 B1 HOLDOUT EXECUTION, BEFORE B2 OUTCOME INSPECTION  
**Protocol tree SHA-256:** `40f1912ec2a95a91891400efdf0444e7aad67b444a19898b4d7294b5c198f28a`  
**Preregistration SHA-256:** `ffccd7789cda190e10e7540f604f935d883faebecb1818dd20df9ec68d333da7`  
**B1 result SHA-256:** `3eadb09ad82ac5b5338c9dd26eadac592bd5378fb4139a1b7a0cc65cbdca1d89`  
**Source-provenance SHA-256:** `e69c83562e57ca645179243d6ef65683c4036ef28c4e3213afd1dc5ff9eb2177`  
**Remote immutable result commit:** `c8dd27cc110eb68de7c80a70d16d48376b0e613f`

## Verdict

B1 is **CONFIRMED AS STRONG PROVISIONAL EVIDENCE** that the frozen historical-state residual correction and validation-only forecastability router contain out-of-sample predictive information beyond the EIA-930 operator day-ahead demand forecast for this four-BA experiment.

This is **not yet a highest-authority historical forecasting claim** because the fixed EIA archive used for the experiment was created on 2025-02-17 and is explicitly classified `V1_PROVISIONAL_REVISION_SENSITIVE`. The result therefore supports the model architecture and routing hypothesis, while the magnitude remains subject to row-level source-vintage/revision sensitivity analysis.

## Holdout result

- Raw selected EIA rows, 2020–2024: **175,392**
- Quarantined rows: **306**
- Training target rows, 2020–2022: **52,538**
- Validation target rows, 2023: **17,506**
- Untouched confirmatory target rows, 2024: **17,483**
- EIA operator baseline 2024 MAE: **1,725.001 MW**
- Frozen routed B1 2024 MAE: **1,272.967 MW**
- Aggregate relative MAE improvement: **26.205%**
- BA-week routed relative-lift bootstrap mean: **17.180%**
- 95% BA-week bootstrap interval: **[14.552%, 19.821%]**
- BA-week clusters: **212**

## Validation-frozen router and 2024 confirmation

| BA | 2023 router decision | 2023 bootstrap lower bound | 2024 routed MAE improvement | 2024 BA-week bootstrap 95% CI |
|---|---:|---:|---:|---:|
| CISO | Promote | +39.248% | +39.980% | [+37.580%, +43.932%] |
| PJM | Promote | +27.156% | +28.816% | [+24.747%, +31.119%] |
| ERCO | Fall back | -1.759% | 0.000% | [0.000%, 0.000%] |
| ISNE | Fall back | -2.387% | 0.000% | [0.000%, 0.000%] |

The router succeeded in the role it was designed for: it promoted CISO and PJM, both retained large positive out-of-sample lift, and it protected the system from the substantial 2024 degradation that the enhanced stack would have produced in ERCO (-5.741%). The ISNE fallback was conservative: the hindsight stack would have improved 2024 MAE by 6.696%, but 2023 did not justify promotion, so the protocol correctly refused that hindsight gain.

## Model-level observations

The ridge residual model alone had the best aggregate 2024 MAE among the fixed candidate models, improving MAE by **28.020%**. The validation-frozen stack improved MAE by **25.394%**. This does not authorize replacing the stack with ridge after seeing 2024. Any such model-selection change belongs to a separately preregistered future version.

The fitted validation-only stack weights were approximately: operator 0.000000000000377, ridge 0.445130, HGB 0.554870. Ridge alpha was selected using 2023 only and froze at **100.0**.

## Interpretation limits

1. The experiment used raw reported `Demand (MW)` and `Demand Forecast (MW)` fields from a single hash-verified fixed PUDL/Zenodo EIA-930 vintage.
2. That vintage post-dates the 2024 target period; prior metadata work established that EIA-930 historical archive files can change across published vintages.
3. The result remains `V1_PROVISIONAL_REVISION_SENSITIVE` until value-level vintage comparisons establish how much the raw demand and forecast series changed and whether model ranking/lift is stable across reasonable vintages.
4. No 2025 replication data was opened in this B1 execution.
5. No HRRR/weather data was used in B1.

## Frozen next test

Proceed to **B2 2024** exactly as preregistered: add only point-in-time-eligible operational HRRR temperature/dewpoint-derived weather state to B1 and evaluate incremental lift over B1. Do not change B1 models, stack weights, router rule, geography, target window, or B2 promotion gate after this verdict.