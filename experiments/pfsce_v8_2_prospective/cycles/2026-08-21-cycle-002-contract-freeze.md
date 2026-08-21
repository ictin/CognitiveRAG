# PFSCE v8.2 Prospective Validation — Cycle 002 Contract Freeze

Cycle ID: `PFSCE-V8.2-PROSPECTIVE-20260821-C002`
Freeze time: `2026-08-21T06:18:00Z`
Methodology: `PFSCE v8.2`
Anchor generation: `PFSCE-v8.2-A0` — OpenAI GPT-5.6 Sol, opaque product-managed checkpoint, strict baseline fallback outside locally validated authority.
Challenger generation: `PFSCE-v8.2-C1-DAGCRITIC` — OpenAI GPT-5.6 Sol, Forecast-DAG/decomposition + critic/coherence, research-only unless separately promoted.

This file freezes new question wording and resolution contracts **before question-directed research**. No baseline probability, PFSCE probability, or answer-directed evidence is frozen in this file. Those are added only in the append-only forecast ledger created after research. Existing cohort packets `GEN-20260820-001` through `GEN-20260820-008` and Cycle-001 packets remain immutable and are not rewritten here.

## External ForecastBench stream

No new external ForecastBench packet is added in this cycle. The currently accessible ForecastBench public surfaces confirm the benchmark is dynamically refreshed, but this run could not establish a fresh, exact unresolved question record with its original benchmark market snapshot and due-date provenance strongly enough to satisfy the prospective integrity requirement. Existing frozen ForecastBench packets remain in the denominator. This is a provenance abstention, not a forecast abstention.

## C002-GEN-1 — broad PFSCE general stream

**Question:** Will the Bank of Japan raise its policy interest rate by at least 25 basis points at its September 2026 monetary-policy meeting?

**Resolution contract:** YES iff the Bank of Japan's official monetary-policy statement for its September 2026 meeting announces an increase of at least 0.25 percentage points in the Bank's target/policy rate relative to the rate immediately before that meeting. Otherwise NO. Primary source: Bank of Japan official monetary-policy decision/statement. If the meeting is postponed beyond September, resolve NO for this contract.

**Forecastability cell:** `macro|central_bank_rate_change|~4w|institutional|strategic-data-dependent|Japan`

**Driver cluster:** `GLOBAL_MONETARY_POLICY_ASIA`. Treat as partially dependent on global energy/inflation shocks but distinct from the frozen FOMC/ECB questions.

## C002-NECF-1 — NECF dense material/grid stream

**Question:** Will ERCOT system-wide demand exceed 85,000 MW in any reported hourly interval from August 22 through August 31, 2026 inclusive?

**Resolution contract:** YES iff ERCOT's official system-demand/load data show system-wide demand strictly greater than 85,000 MW for at least one hourly interval whose local operating date is 2026-08-22 through 2026-08-31 inclusive. Otherwise NO. Primary source: ERCOT official load/demand data; EIA-930 may be used only as a secondary audit source if ERCOT's official archive is ambiguous. Missing-data periods do not count as NO until the reporting window has closed and official data are available.

**Forecastability cell:** `electricity|ERCOT_extreme_load_threshold|1-10d|ISO|weather-sensitive-material`

**Driver cluster:** `US_GRID_HEAT_SOUTH`. Dependence with the existing PJM/CAISO heat packets must be reflected in effective-N accounting.

## C002-AOM-1 — AOM-directed scientifically isolated stream

**Question:** Will Bitcoin trade at or above USD 130,000 on Coinbase BTC-USD before September 30, 2026 23:59:59 UTC?

**Resolution contract:** YES iff the high price of any completed 1-minute Coinbase BTC-USD candle is at least USD 130,000 between the freeze time and 2026-09-30T23:59:59Z. Otherwise NO. Primary source: Coinbase official BTC-USD historical market data/API; a second institutional market-data source may audit a disputed print but may not replace the primary contract.

**Forecastability cell:** `crypto|price_threshold|~40d|BTC|reflexive-market|global`

**Driver cluster:** `CRYPTO_RISK_ASSETS`. This stream is scientifically isolated from broad promotion claims because the question was selected for economic relevance.

## Integrity rules for Cycle 002

1. Research begins only after this contract-freeze commit exists.
2. Baselines must be contemporaneous snapshots available after freeze but before the first forecast issue is persisted.
3. Existing first-issue probabilities and baseline snapshots are never rewritten.
4. All new raw/anchor/challenger probabilities are explicitly uncalibrated unless a local calibration passport exists.
5. A0 falls back to the strongest frozen baseline when the exact forecastability cell lacks promoted authority.
6. Full denominator is preserved, including provenance exclusions, baseline fallback, and abstentions.
7. No new question may be deleted later because it performs badly.
