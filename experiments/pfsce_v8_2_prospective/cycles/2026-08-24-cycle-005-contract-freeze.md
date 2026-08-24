# PFSCE v8.2 Prospective Cycle 005 — Contract Freeze

Frozen before answer-directed research on 2026-08-24 under program `PFSCE-V8.2-PROSPECTIVE-20260820`.

## Immutable parent cohorts
Maintain `PFSCE-GEN-20260820-v1` and `PFSCE-GEN-20260820-v1-B` exactly as issued. Original baseline snapshots, first-issue PFSCE probabilities, resolution contracts, cohort membership, and prior trajectory entries must not be rewritten. Any correction must be appended with an explicit integrity flag.

## Model generations
- Anchor: `PFSCE-v8.2-A0`; provider OpenAI; model GPT-5.6 Sol; checkpoint `opaque_product_managed`; policy = baseline fallback outside locally validated authority.
- Challenger: `PFSCE-v8.2-C1-DAGCRITIC`; provider OpenAI; model GPT-5.6 Sol; checkpoint `opaque_product_managed`; research-only Forecast DAG/decomposition + critic + coherence challenger.

## Existing cohort review rule
For GEN-001 through GEN-008, search only for evidence whose `availability_time` is later than the 2026-08-23T06:30:00Z Cycle-004 cutoff. Resolution is allowed only if the frozen resolution contract is already satisfied by an admissible primary source. Otherwise append a probability update only if genuinely new evidence is material; else record `NO_UPDATE`.

## New-question selection streams and pre-research contracts
Candidate admission is optional; provenance/independence gates may exclude any candidate after research. Exclusions remain recorded in the denominator-selection audit.

### Stream 1 — External ForecastBench
Selection rule: inspect the latest available ForecastBench unresolved set for a question not previously used in this program, with a frozen pre-issue market/baseline snapshot and auditable due date. Do not admit if provenance cannot be independently verified. Resolution = ForecastBench's own frozen resolution criteria/source.

### Stream 2 — Broad PFSCE general
Preselected candidate `C005-GEN-1`: **Will NASA's Artemis II mission launch on or before 2026-09-30 23:59:59 UTC?**
Resolution: YES only if NASA's official mission/launch record confirms liftoff by the deadline; NO otherwise after the deadline. Baseline must be frozen from the strongest contemporaneous contract-matched market/crowd source found after this freeze; if none exists, the candidate may be excluded rather than assigned an invented baseline.
Forecastability cell: `spaceflight|mission_launch_deadline|~5w|NASA|operational-schedule|US`.
Episode cluster: `SPACEFLIGHT_OPERATIONS_US`.

### Stream 3 — NECF dense material/grid
Selection rule: do not duplicate the separately running NECF confirmatory A0/C1/W1 live stream. Admit a new grid packet only if it uses a distinct target definition/horizon with a strong frozen comparator and adds material effective independent N. Otherwise record `SEPARATE_CONFIRMATORY_STREAM_AVOIDS_DUPLICATION`.

### Stream 4 — AOM-directed, scientifically isolated
Preselected candidate `C005-AOM-1`: **Will NVIDIA (NVDA) close at or above USD 200 on any regular U.S. trading day on or before 2026-09-30?**
Resolution: YES if an authoritative market-price source records an official regular-session close >=200 by the deadline; NO otherwise. Baseline must be a contemporaneous contract-matched market probability if available; weak proxies must be explicitly labeled and do not qualify for general PFSCE promotion.
Forecastability cell: `equity|price_threshold|~5w|NVDA|reflexive-market|US`.
Episode cluster: `AI_EQUITY_RISK_ASSETS`.
Scientific isolation: AOM selection cannot contribute to broad-general promotion claims.

## Admission / scoring rules
Every admitted packet must preserve: strongest frozen baseline and source time; raw A0 and C1 probabilities; routed A0 probability and router action; calibrated probability if any; probability provenance; authority grade; Module Passport/MEG; epistemic uncertainty range; evidence lineages and effective evidence N; update triggers; prior trajectory; exposure/intervention status; resolution contract reference; model/version metadata.

No model promotion is allowed without local prospective authority. Preserve baseline fallbacks and abstentions in the denominator. No retrospective outcome knowledge, future-aware evidence, post-outcome question selection, or post-cutoff baseline substitution is allowed.
