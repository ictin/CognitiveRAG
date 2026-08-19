# B2 HRRR source-access adapter freeze

Status: **FROZEN BEFORE B2 MODEL FITTING OR 2024 B2 OUTCOME SCORING**

Protocol: `NECF-001-WP2-CONFIRMATORY-2024-v1`

The preregistered scientific source remains archived operational NOAA HRRR and the canonical availability gate remains the original NOAA NODD bucket `noaa-hrrr-bdp-pds`.

## Access adaptation

Full-CONUS GRIB2 messages are not an efficient way to extract 16 fixed point locations across five years. Before any B2 model is fitted or any B2 holdout score is inspected, this implementation freezes the following source-access adaptation:

1. Use `noaa-hrrr-bdp-pds` object metadata as the **canonical point-in-time availability authority**. A 12Z run is eligible for a daily origin only when every HRRR GRIB2 forecast object required for that origin's 12 target valid hours has `LastModified <= forecast_origin`.
2. Read the corresponding temperature and dew-point numeric values from the University of Utah/MesoWest `hrrrzarr` archive, an access-optimized Zarr transformation of HRRR model output listed alongside the NOAA HRRR Open Data resource.
3. The Zarr mirror is treated as a **transport/representation layer**, not as a new predictive source. No reanalysis, later observations, alternate weather model, or future-aware field is admitted.
4. Variables remain exactly the frozen primitive fields: 2-m TMP and 2-m DPT. Geography remains the frozen four equal-weight load centres per BA. Derived fields remain temperature, dew point, RH, heat index and cooling-degree-hours over 18C.
5. Spatial lookup is deterministic nearest-HRRR-grid-point using the documented HRRR Lambert Conformal grid. No geography tuning is allowed after B2 scoring.
6. Forecast lead `fxx` maps to Zarr forecast index `fxx-1`, because the forecast arrays contain non-zero leads F01...FXX.
7. Zarr dtype/compression/shape are read from each array's `.zarray` metadata rather than hard-coded, because the archive changed default surface-variable precision during 2024.
8. If required HRRR values are absent or the canonical NOAA object-availability gate fails, the weather row is missing. No retrospective weather observation or later run substitutes for it.

## Authority caveat

Because `hrrrzarr` is a derived representation and historically used reduced floating-point precision for many surface fields, B2 should be labeled `POINT_IN_TIME_ELIGIBLE_DERIVED_HRRR_MIRROR` rather than claiming byte-identical GRIB2 authority. A successful B2 must still pass the preregistered incremental gate over frozen B1; this access adaptation does not lower that gate.

## Preflight requirement

Before full feature extraction, a metadata-only/source-integrity preflight must confirm:
- original NOAA objects exist and their timestamps satisfy the origin rule on fixed sample dates;
- required Zarr arrays/chunks exist;
- forecast lead indexing is in range;
- extracted TMP/DPT values are finite and physically plausible;
- metadata correctly detects the archive dtype transition.

No B2 demand outcomes may be scored until this preflight passes.