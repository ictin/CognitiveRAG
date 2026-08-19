from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

import run_b1 as b1

AUDIT_ID = "NECF-001-EIA930-REVISION-AUDIT-v1"
ROW_RECORDS = [
    "14881638", "14949257", "15568995", "15877505", "17241309",
    "17846570", "18448416", "19367770", "19787334",
]
FULL_PIPELINE_RECORDS = ["14881638", "17241309", "18448416", "19787334"]
YEARS = list(range(2020, 2025))
SEED = 20260819
BOOT = 5000
BAS = list(b1.BAS)


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def request_json(url: str) -> dict:
    for i in range(7):
        try:
            r = requests.get(url, timeout=(30, 120), headers={"User-Agent": "PFSCE-NECF-revision-audit/1.0"})
            if r.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == 6:
                raise
            time.sleep(min(60, 2 ** (i + 1)))
    raise AssertionError


def record_files(record_id: str) -> tuple[dict[str, dict], dict]:
    meta = request_json(f"https://zenodo.org/api/records/{record_id}")
    files = {}
    for f in meta.get("files", []):
        key = f.get("key")
        if key:
            files[key] = f
    return files, {
        "record_id": record_id,
        "created": meta.get("created"),
        "updated": meta.get("updated"),
        "publication_date": (meta.get("metadata") or {}).get("publication_date"),
        "version": (meta.get("metadata") or {}).get("version"),
        "doi": (meta.get("metadata") or {}).get("doi"),
    }


def download_record_file(record_id: str, filename: str, cache: Path) -> tuple[Path, dict]:
    files, record_meta = record_files(record_id)
    if filename not in files:
        raise RuntimeError(f"{record_id} missing {filename}")
    fm = files[filename]
    url = (fm.get("links") or {}).get("content") or f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
    p = cache / record_id / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    checksum = str(fm.get("checksum") or "")
    expected_algo, expected_hex = (checksum.split(":", 1) + [""])[:2] if ":" in checksum else ("", "")
    if not p.exists() or (expected_algo and digest(p, expected_algo) != expected_hex):
        if p.exists():
            p.unlink()
        for i in range(7):
            try:
                with requests.get(url, stream=True, timeout=(30, 240), headers={"User-Agent": "PFSCE-NECF-revision-audit/1.0"}) as r:
                    if r.status_code in {429, 500, 502, 503, 504}:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    tmp = p.with_suffix(p.suffix + ".part")
                    with tmp.open("wb") as w:
                        for c in r.iter_content(1024 * 1024):
                            if c:
                                w.write(c)
                    tmp.replace(p)
                    break
            except Exception:
                if i == 6:
                    raise
                time.sleep(min(60, 2 ** (i + 1)))
    got = digest(p, expected_algo or "sha256")
    if expected_algo and got != expected_hex:
        raise RuntimeError(f"Checksum mismatch {record_id}/{filename}: expected={checksum} got={got}")
    return p, {
        **record_meta,
        "filename": filename,
        "expected_checksum": checksum,
        "verified_checksum": f"{expected_algo}:{got}" if expected_algo else None,
        "sha256": digest(p, "sha256"),
        "size": p.stat().st_size,
        "url": url,
    }


def half_filenames(year: int) -> list[str]:
    return [f"eia930-{year}half1.zip", f"eia930-{year}half2.zip"]


def load_record_years(record_id: str, years: list[int], cache: Path) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    provenance = []
    for year in years:
        for fn in half_filenames(year):
            print("DOWNLOAD", record_id, fn, flush=True)
            p, prov = download_record_file(record_id, fn, cache)
            d, members = b1.read_zip(p)
            lo = pd.Timestamp(f"{year}-01-01T00:00:00Z")
            hi = pd.Timestamp(f"{year}-12-31T23:59:59Z")
            d = d[(d.datetime_utc >= lo) & (d.datetime_utc <= hi)].copy()
            frames.append(d)
            provenance.append({**prov, "selected_rows": int(len(d)), "members": members})
    x = pd.concat(frames, ignore_index=True)
    x = x.sort_values(["ba", "datetime_utc"]).drop_duplicates(["ba", "datetime_utc"], keep="last")
    return x, provenance


def changed_mask(a: pd.Series, b: pd.Series) -> np.ndarray:
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    return ~np.isclose(av, bv, rtol=0, atol=0, equal_nan=True)


def delta_stats(x: pd.Series) -> dict:
    a = np.abs(pd.to_numeric(x, errors="coerce").dropna().to_numpy(float))
    if len(a) == 0:
        return {"n": 0, "mean_abs": None, "median_abs": None, "p95_abs": None, "max_abs": None}
    return {
        "n": int(len(a)),
        "mean_abs": float(a.mean()),
        "median_abs": float(np.median(a)),
        "p95_abs": float(np.quantile(a, 0.95)),
        "max_abs": float(a.max()),
    }


def compare_to_original(original: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    keys = ["ba", "datetime_utc"]
    m = original.merge(candidate, on=keys, how="outer", suffixes=("_orig", "_cand"), indicator=True)
    both = m[m._merge == "both"].copy()
    demand_changed = changed_mask(both.demand_orig, both.demand_cand)
    forecast_changed = changed_mask(both.forecast_orig, both.forecast_cand)
    both["demand_delta"] = both.demand_cand - both.demand_orig
    both["forecast_delta"] = both.forecast_cand - both.forecast_orig

    def subset_stats(g: pd.DataFrame) -> dict:
        dc = changed_mask(g.demand_orig, g.demand_cand)
        fc = changed_mask(g.forecast_orig, g.forecast_cand)
        return {
            "common_rows": int(len(g)),
            "demand_changed_rows": int(dc.sum()),
            "forecast_changed_rows": int(fc.sum()),
            "changed_either_rows": int((dc | fc).sum()),
            "demand_changed_fraction": float(dc.mean()) if len(g) else None,
            "forecast_changed_fraction": float(fc.mean()) if len(g) else None,
            "demand_delta": delta_stats(g.loc[dc, "demand_delta"]),
            "forecast_delta": delta_stats(g.loc[fc, "forecast_delta"]),
        }

    out = subset_stats(both)
    out["rows_only_original"] = int((m._merge == "left_only").sum())
    out["rows_only_candidate"] = int((m._merge == "right_only").sum())
    out["by_ba"] = {str(ba): subset_stats(g) for ba, g in both.groupby("ba")}
    return out


def fit_stack(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = pred.shape[1]
    fun = lambda w: np.mean(np.abs(y - pred @ w))
    r = minimize(fun, np.repeat(1 / n, n), method="SLSQP", bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    if not r.success:
        raise RuntimeError(r.message)
    return r.x


def routed_pipeline(raw: pd.DataFrame) -> dict:
    good, bad = b1.quarantine(raw)
    f = b1.target_filter(b1.features(good))
    F = b1.cols(f)
    train = b1.split(f, "2020-01-01T00:00:00Z", "2022-12-31T23:00:00Z").dropna(subset=["demand", "forecast", "residual"])
    val = b1.split(f, "2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z").dropna(subset=["demand", "forecast", "residual"])
    test = b1.split(f, "2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z").dropna(subset=["demand", "forecast", "residual"])
    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError(f"Empty split train={len(train)} val={len(val)} test={len(test)}")

    ridge, ridge_meta = b1.ridge_fit(train[F], train.residual, val[F], val.residual)
    hgb = b1.hgb_fit(train[F], train.residual)
    val = val.copy()
    vr = val.forecast.to_numpy() + ridge.predict(val[F])
    vh = val.forecast.to_numpy() + hgb.predict(val[F])
    weights = fit_stack(np.column_stack([val.forecast.to_numpy(), vr, vh]), val.demand.to_numpy())
    val["pred_stack"] = np.column_stack([val.forecast.to_numpy(), vr, vh]) @ weights

    router = {}
    weekly_val = b1.weekly_lifts(val, "pred_stack")
    for ba in BAS:
        ci = b1.boot([v for bb, _, v in weekly_val if bb == ba])
        router[ba] = {"promoted": bool(ci["lower"] is not None and ci["lower"] > 0), "validation_bootstrap": ci}

    test = test.copy()
    tr = test.forecast.to_numpy() + ridge.predict(test[F])
    th = test.forecast.to_numpy() + hgb.predict(test[F])
    ts = np.column_stack([test.forecast.to_numpy(), tr, th]) @ weights
    test["pred_stack"] = ts
    test["pred_routed"] = np.where(test.ba_code.map({k: v["promoted"] for k, v in router.items()}), test.pred_stack, test.forecast)

    weekly = b1.weekly_lifts(test, "pred_routed")
    boot = b1.boot([v for _, _, v in weekly])
    operator = b1.metrics(test.demand, test.forecast)
    routed = b1.metrics(test.demand, test.pred_routed, test.forecast)
    by_ba = {}
    for ba, g in test.groupby("ba_code"):
        by_ba[str(ba)] = {
            "router_promoted": bool(router[str(ba)]["promoted"]),
            "operator": b1.metrics(g.demand, g.forecast),
            "routed": b1.metrics(g.demand, g.pred_routed, g.forecast),
        }
    return {
        "n_train": int(len(train)),
        "n_validation": int(len(val)),
        "n_test": int(len(test)),
        "n_quarantined": int(len(bad)),
        "ridge": ridge_meta,
        "stack_weights": [float(x) for x in weights],
        "router": router,
        "operator": operator,
        "routed": routed,
        "BA_week_bootstrap": boot,
        "by_ba": by_ba,
    }


def main() -> None:
    out = Path("experiment_output/eia_revision_audit")
    cache = out / "raw"
    out.mkdir(parents=True, exist_ok=True)

    # Row-level 2024 comparison across a denser sequence of archive vintages.
    row_data = {}
    row_provenance = {}
    for rid in ROW_RECORDS:
        d, prov = load_record_years(rid, [2024], cache)
        row_data[rid] = d
        row_provenance[rid] = prov
    original = row_data[ROW_RECORDS[0]]
    row_compare = {rid: compare_to_original(original, row_data[rid]) for rid in ROW_RECORDS[1:]}

    # Full frozen architecture reruns over a spanning set of vintages.
    full_results = {}
    full_provenance = {}
    for rid in FULL_PIPELINE_RECORDS:
        d, prov = load_record_years(rid, YEARS, cache)
        full_provenance[rid] = prov
        print("FIT_FULL_PIPELINE", rid, flush=True)
        full_results[rid] = routed_pipeline(d)

    all_strong = True
    any_fragile = False
    lifts = []
    routers = {}
    for rid, r in full_results.items():
        lift = float(r["routed"]["relative_mae_improvement"])
        lower = r["BA_week_bootstrap"]["lower"]
        lifts.append(lift)
        routers[rid] = {ba: bool(v["promoted"]) for ba, v in r["router"].items()}
        all_strong = all_strong and lift >= 0.10 and lower is not None and lower > 0
        any_fragile = any_fragile or lift <= 0 or lower is None or lower <= 0
    if all_strong:
        verdict = "ROBUST_TO_REVISION"
    elif any_fragile:
        verdict = "FRAGILE_TO_REVISION"
    else:
        verdict = "REVISION_SENSITIVE_BUT_DIRECTIONALLY_ROBUST"

    result = {
        "audit_id": AUDIT_ID,
        "status": "FROZEN_REVISION_SENSITIVITY_AUDIT_RESULT",
        "authority": "DIAGNOSTIC_ONLY_NOT_POINT_IN_TIME_AUTHORITY",
        "row_level_2024": {
            "original_record": ROW_RECORDS[0],
            "records": ROW_RECORDS,
            "comparisons_vs_original": row_compare,
            "provenance": row_provenance,
        },
        "full_pipeline_2020_2024": {
            "records": FULL_PIPELINE_RECORDS,
            "results": full_results,
            "provenance": full_provenance,
            "routed_lift_min": float(min(lifts)),
            "routed_lift_max": float(max(lifts)),
            "routed_lift_spread_percentage_points": float((max(lifts) - min(lifts)) * 100),
            "router_decisions": routers,
        },
        "frozen_interpretation": {
            "verdict": verdict,
            "all_full_pipeline_vintages_routed_lift_ge_10pct_and_bootstrap_lower_gt_zero": bool(all_strong),
            "any_full_pipeline_vintage_nonpositive_or_CI_crosses_zero": bool(any_fragile),
            "note": "Even ROBUST_TO_REVISION remains V1_PROVISIONAL_REVISION_SENSITIVE and cannot establish point-in-time historical authority.",
        },
    }

    rp = out / "EIA930_REVISION_AUDIT.json"
    rp.write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = [
        "# EIA-930 Revision Sensitivity Audit",
        "",
        f"Audit: `{AUDIT_ID}`",
        "",
        f"**Frozen verdict: {verdict}.**",
        "",
        f"Full-pipeline routed-lift range: **{min(lifts)*100:.3f}% to {max(lifts)*100:.3f}%**",
        f"Spread: **{(max(lifts)-min(lifts))*100:.3f} percentage points**",
        "",
        "## Full-pipeline vintages",
        "",
    ]
    for rid in FULL_PIPELINE_RECORDS:
        r = full_results[rid]
        lines.append(
            f"- `{rid}`: routed lift **{r['routed']['relative_mae_improvement']*100:.3f}%**, "
            f"BA-week CI **[{r['BA_week_bootstrap']['lower']*100:.3f}%, {r['BA_week_bootstrap']['upper']*100:.3f}%]**, "
            f"router `{routers[rid]}`."
        )
    lines += [
        "",
        "## Authority implication",
        "",
        "This audit measures sensitivity to revised archives. It does not reconstruct what exact EIA values were available at each historical forecast origin. Therefore a robust result strengthens the component claim but does not remove the `V1_PROVISIONAL_REVISION_SENSITIVE` label.",
    ]
    (out / "EIA930_REVISION_AUDIT_VERDICT.md").write_text("\n".join(lines) + "\n")
    checksums = {p.name: digest(p, "sha256") for p in out.iterdir() if p.is_file()}
    (out / "SHA256.json").write_text(json.dumps(checksums, indent=2, sort_keys=True))
    print("PFSCE_EIA_REVISION_AUDIT_BEGIN")
    print(json.dumps({
        "audit_id": AUDIT_ID,
        "verdict": verdict,
        "lift_min": min(lifts),
        "lift_max": max(lifts),
        "lift_spread_pp": (max(lifts)-min(lifts))*100,
        "routers": routers,
        "row_change_summary": {
            rid: {
                "demand_changed_fraction": row_compare[rid]["demand_changed_fraction"],
                "forecast_changed_fraction": row_compare[rid]["forecast_changed_fraction"],
                "changed_either_rows": row_compare[rid]["changed_either_rows"],
            } for rid in row_compare
        },
    }, indent=2))
    print("PFSCE_EIA_REVISION_AUDIT_END")


if __name__ == "__main__":
    main()
