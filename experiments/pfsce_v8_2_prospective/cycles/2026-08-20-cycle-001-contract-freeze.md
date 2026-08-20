# PFSCE v8.2 Prospective Cycle 001 — Contract Freeze

Cycle ID: `PFSCE-V8.2-PROSPECTIVE-20260820-C001`
Contract state: **FROZEN BEFORE QUESTION-SPECIFIC RESEARCH FOR THIS COHORT**
Evidence cutoff target: 2026-08-20 09:00 Europe/Amsterdam, with actual per-source `availability_time` recorded later.

## Stable generations

- Anchor: `PFSCE-v8.2-A0`, methodology PFSCE v8.2 production-eligible core only; model `OpenAI GPT-5.6 Sol`; product checkpoint opaque/product-managed; probability output must preserve baseline/fallback/abstention policy.
- Challenger: `PFSCE-v8.2-C1-DAGCRITIC`, same model family/checkpoint exposure as available in this run, plus explicit Forecast DAG/decomposition, independent critic pass, probability coherence check, and locally justified adjustment only. Challenger cannot overwrite A0.

## Frozen cohort and resolution contracts

### Stream 1 — External ForecastBench

**FB-E1** — ForecastBench ID `s1FQwmAZ87EGxUsRqQKQ`: *Will Salt Lake City get a Major League Baseball team before 2030?*
- Inherit the official ForecastBench/Manifold resolution contract verbatim from the 2026-08-16 question set.
- YES if MLB officially announces a team based in Salt Lake City before 2029-12-31 23:59 local market deadline under the source contract; home game may occur later as specified by the source.
- Baseline candidate frozen from ForecastBench question set: market value `0.2713279326612697` at `2026-08-06T00:00:00Z`.

**FB-E2** — ForecastBench ID `0N8CI06Shc`: *Will the US attack Cuba before the 2026 midterm elections?*
- Inherit the official ForecastBench/Manifold resolution contract from the 2026-08-16 question set.
- YES for an overt US military drone/missile strike, sustained bombing, or invasion before the 2026-11-03 midterm election cutoff; covert sabotage/kidnapping/proxy action does not count unless it becomes overt US-banner war, per source clarification.
- Baseline candidate frozen from ForecastBench question set: market value `0.10197923801761948` at `2026-08-06T00:00:00Z`.

### Stream 2 — Broad PFSCE general, economic attractiveness ignored

**GEN-1** — *Will Copernicus report September 2026 global surface-air temperature at least 1.50°C above the 1850–1900 pre-industrial average?*
- YES iff the Copernicus Climate Change Service monthly climate bulletin for September 2026 explicitly gives the September global surface-air temperature anomaly relative to 1850–1900 as `>= 1.50°C` (using its stated conversion/method for the month).
- NO iff the corresponding published value is `< 1.50°C`.
- If Copernicus materially changes the reference methodology before publication, preserve the originally reported September-2026 figure if a directly comparable 1850–1900 anomaly is provided; otherwise mark resolution ambiguous rather than reinterpret.

**GEN-2** — *Will the 2026 Nobel Prize in Physics be awarded to exactly three laureates?*
- YES iff the Nobel Prize official 2026 Physics announcement lists exactly three laureates.
- NO iff it lists any other number of laureates.
- Resolution source: NobelPrize.org official prize announcement/press release.

### Stream 3 — NECF dense material/grid

**NECF-1** — *Will PJM RTO system load exceed 160,000 MW in any hourly interval from 2026-09-01 00:00 through 2026-09-15 23:59 EPT?*
- YES iff PJM Data Miner / official PJM hourly RTO load reports any hour in that window with system load `>160,000 MW`.
- Use the latest operationally accepted metered/settled value available after the window closes; preserve vintage metadata. Do not silently substitute an ex-post revised series without recording revision time.

**NECF-2** — *Will CAISO system demand exceed 48,000 MW on a 5-minute-average basis at least once from 2026-09-01 00:00 through 2026-09-15 23:59 Pacific Time?*
- YES iff official CAISO/OASIS or Today's Outlook archived demand data contain a 5-minute system-demand average `>48,000 MW` in the window.
- Record data vintage and revision time used at resolution.

### Stream 4 — AOM-directed, scientifically isolated from general-skill claims

**AOM-1** — *Will ETH/USDT trade at or above $5,000 on Binance in any final 1-minute candle high before 2026-12-31 23:59 ET?*
- YES iff any Binance ETH/USDT 1-minute candle with timestamp no later than the deadline has final recorded `High >= 5000`.
- NO otherwise.
- Resolution source: Binance ETH/USDT 1-minute candles; record data retrieval/vintage.

**AOM-2** — *Will Apple publicly announce a commercially intended foldable iPhone before 2027?*
- YES iff before 2027-01-01 00:00 Pacific Time Apple publishes an official Newsroom/product announcement explicitly presenting an iPhone with a foldable/folding display intended for commercial sale, preorder, or customer release.
- Patents, prototypes, supply-chain rumors, analyst claims, or third-party demonstrations do not count.
- NO if no qualifying Apple announcement occurs by the deadline.

## Denominator / dependence freeze

Nominal N = 8 questions. No question may be deleted because the router abstains or falls back.
Initial latent-driver clusters: `MLB_EXPANSION` (FB-E1), `US_CUBA_SECURITY` (FB-E2), `GLOBAL_CLIMATE` (GEN-1), `NOBEL_SELECTION` (GEN-2), `US_GRID_HEAT` (NECF-1 and NECF-2; correlated cluster, effective N contribution <= 1.5 jointly), `CRYPTO_RISK_ASSET` (AOM-1), `APPLE_PRODUCT_CYCLE` (AOM-2). Initial effective independent N ceiling = 7.5 before any later cross-cluster dependence adjustment.

## Integrity note

Several other questions were inspected while scoping this run (including personalized cancer mRNA approval, 2026 House control, Arctic sea ice, Atlantic named-storm counts, GPT-6 timing, and BTC $150k). They are deliberately excluded from this pristine cycle and must not be retroactively added as Cycle-001 observations.
