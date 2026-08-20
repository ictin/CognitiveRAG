# PFSCE v8.2 NECF Live Source/Target Selection Clarification

Protocol: `PFSCE-V8.2-NECF-PROSPECTIVE-20260820-v1`
Status: **FROZEN BEFORE FIRST 2026-08-20 10:30 America/New_York issue time**
Purpose: remove implementation ambiguity without changing any model, feature, router, promotion gate, or outcome-dependent choice.

## 1. Target-hour semantics

The EIA-930 historical files used by the frozen B1 implementation label the UTC timestamp at the **end of the reported hour**. Therefore, for an issue origin of 10:30 America/New_York, the 12 eligible target intervals are the 12 hourly intervals whose end timestamps are the first twelve whole-hour timestamps strictly after the origin.

For 2026-08-20 (EDT, UTC-4), this means:
- issue origin = `2026-08-20T10:30:00-04:00` = `2026-08-20T14:30:00Z`;
- target end timestamps = `15:00Z, 16:00Z, ..., 23:00Z, 00:00Z, 01:00Z, 02:00Z` (12 rows per BA), corresponding to 11:00 through 22:00 EDT.

This reproduces the historical `delta > 0 and delta <= 12h` target filter. DST conversion must be done through `America/New_York`, never a hard-coded fixed UTC offset.

## 2. EIA-930 operator baseline snapshot

Canonical live source family: EIA API v2 `electricity/rto/region-data`, Form EIA-930. Required BAs are CISO, ERCO, PJM and ISNE. Required types are demand `D` and day-ahead demand forecast `DF` (or the API's metadata-equivalent demand-forecast type if the live schema exposes a renamed code; any such mapping must be recorded before values are used).

At each issue run:
1. retrieve the live EIA route/facet metadata first and record it;
2. retrieve the required D/DF rows no later than the issue cutoff;
3. persist the exact response bytes or a canonicalized payload plus SHA-256 and retrieval timestamp;
4. use only forecast values present in that frozen pre-cutoff snapshot as the strong baseline;
5. if a required target-hour DF value is absent, record `BASELINE_UNAVAILABLE` for that BA-hour and keep it in the denominator;
6. never fill a missing issue-time baseline from a later EIA retrieval.

The retrieval timestamp is an upper bound proving the data were public no later than that moment. EIA's Form-930 reporting rules state that respondents post hourly demand within 60 minutes and separately post the prior day's demand/forecast package by 07:00 Eastern; the live snapshot is still required because API ingestion/representation can differ from respondent submission time.

## 3. HRRR W1 source selection

The historical B2 source-adapter rule is retained exactly rather than replaced by adaptive cycle selection:

- canonical availability authority: NOAA NODD S3 bucket `noaa-hrrr-bdp-pds` object metadata;
- weather run: **12Z only** for the daily 10:30 America/New_York origin;
- the 12Z run is eligible only if every GRIB2 forecast object required for all 12 target valid hours has `LastModified <= forecast_origin`;
- if that condition fails, W1 weather is missing/abstains for the affected row/day; **no 06Z, later HRRR cycle, observation, reanalysis or alternate weather model is substituted**;
- numeric TMP/DPT values may continue to be read from the University of Utah/MesoWest `hrrrzarr` representation only after NOAA-object availability passes, preserving the historical `POINT_IN_TIME_ELIGIBLE_DERIVED_HRRR_MIRROR` authority label.

For the 2026-08-20 origin above, the required 12Z valid hours correspond to HRRR leads F03 through F14 inclusive. All twelve canonical NOAA objects must satisfy the availability gate.

## 4. Snapshot and append-only paths

Every live issue day must preserve immutable artifacts under:
`experiments/pfsce_v8_2_prospective/necf_live/YYYY-MM-DD/`

Minimum artifacts:
- `issue_manifest.json` — issue origin, protocol/code/model generation, target hours, router states;
- `eia_issue_snapshot.json` or equivalent canonical payload + hash;
- `hrrr_availability.json` — required object keys, LastModified timestamps and eligibility result;
- `predictions.csv`/`.json` — operator, A0, C1, W1 values and reason codes for fallback/abstention;
- `integrity.json` — 15 frozen first-run audit checks;
- later `truth_first_published.json` and revision records, never overwriting the issue artifacts.

## 5. No-authority-change statement

This clarification only fixes timestamp/source-selection semantics that were implicit in the historical executable and B2 adapter. It does not use any 2026-08-20 target outcomes and does not alter A0/C1/W1 feature sets, model parameters, router decisions or promotion thresholds.
