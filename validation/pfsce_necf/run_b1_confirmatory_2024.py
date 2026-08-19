from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROTOCOL_ID = "NECF-001-WP2-CONFIRMATORY-2024-v1"
PROTOCOL_TREE_SHA256 = "40f1912ec2a95a91891400efdf0444e7aad67b444a19898b4d7294b5c198f28a"
EBA_URL = "https://www.eia.gov/opendata/bulk/EBA.zip"
BAS = ["CISO", "ERCO", "PJM", "ISNE"]
TARGET_SERIES = {f"EBA.{ba}-ALL.D.H": (ba, "demand") for ba in BAS} | {f"EBA.{ba}-ALL.DF.H": (ba, "forecast") for ba in BAS}
TRAIN_START = pd.Timestamp("2020-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2022-12-31T23:59:59Z")
VAL_START = pd.Timestamp("2023-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2023-12-31T23:59:59Z")
TEST_START = pd.Timestamp("2024-01-01T00:00:00Z")
TEST_END = pd.Timestamp("2024-12-31T23:59:59Z")
ALPHAS = [0.1, 1.0, 10.0, 100.0]
SEED = 20260819
N_BOOT = 5000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "PFSCE-NECF-confirmatory-validation/1.0"}
    with requests.get(url, stream=True, timeout=(30, 300), headers=headers) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        return {"url": url, "status": r.status_code, "etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified"), "content_length_header": r.headers.get("Content-Length"), "downloaded_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def parse_eia_time(v: object):
    if v is None:
        return pd.NaT
    s = str(v).strip()
    t = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(t) and len(s) >= 11:
        return pd.NaT
    return t


def extract_series(zip_path: Path):
    found = {}
    with zipfile.ZipFile(zip_path) as zf:
        txts = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if len(txts) != 1:
            raise RuntimeError(f"Expected exactly one .txt in EBA.zip, found {txts[:10]}")
        member = txts[0]
        with zf.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            for line in text:
                if '"series_id"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = obj.get("series_id")
                if sid in TARGET_SERIES:
                    found[sid] = obj
                    if len(found) == len(TARGET_SERIES):
                        break
    missing = sorted(set(TARGET_SERIES) - set(found))
    if missing:
        raise RuntimeError(f"Missing target EIA series: {missing}")
    rows = []
    series_meta = {}
    for sid, obj in found.items():
        ba, field = TARGET_SERIES[sid]
        series_meta[sid] = {"name": obj.get("name"), "units": obj.get("units"), "start": obj.get("start"), "end": obj.get("end"), "last_updated": obj.get("last_updated"), "n_points": len(obj.get("data") or [])}
        for pair in obj.get("data") or []:
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            t = parse_eia_time(pair[0])
            if pd.isna(t):
                continue
            try:
                value = float(pair[1]) if pair[1] not in (None, "null", "", "NA") else np.nan
            except (TypeError, ValueError):
                value = np.nan
            rows.append({"datetime_utc": t, "ba": ba, "field": field, "value": value})
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["datetime_utc", "ba"], columns="field", values="value", aggfunc="last").reset_index()
    wide.columns.name = None
    for c in ["demand", "forecast"]:
        if c not in wide:
            wide[c] = np.nan
    wide = wide[["datetime_utc", "ba", "demand", "forecast"]].sort_values(["ba", "datetime_utc"])
    return wide, {"zip_member": member, "series": series_meta}


def quarantine(x):
    ratio = x["demand"] / x["forecast"]
    ok = x["demand"].notna() & x["forecast"].notna() & (x["demand"] > 0) & (x["forecast"] > 0) & ratio.between(0.2, 5.0)
    return x.loc[ok].copy(), x.loc[~ok].copy()


def build_features(x):
    x = x.sort_values(["ba", "datetime_utc"]).copy()
    x["residual"] = x["demand"] - x["forecast"]
    dt = pd.to_datetime(x["datetime_utc"], utc=True)
    x["hour"] = dt.dt.hour
    x["day_of_week"] = dt.dt.dayofweek
    x["month"] = dt.dt.month
    x["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    g = x.groupby("ba", group_keys=False)
    for lag in [48, 72, 168]:
        x[f"residual_lag_{lag}"] = g["residual"].shift(lag)
        x[f"demand_lag_{lag}"] = g["demand"].shift(lag)
    for win in [168, 336, 720]:
        x[f"resid_mean_{win}"] = g["residual"].transform(lambda s: s.shift(48).rolling(win, min_periods=max(24, win // 4)).mean())
        x[f"resid_std_{win}"] = g["residual"].transform(lambda s: s.shift(48).rolling(win, min_periods=max(24, win // 4)).std())
    x = pd.get_dummies(x, columns=["ba"], prefix="ba", dtype=float)
    ba_cols = [c for c in x.columns if c.startswith("ba_")]
    x["ba_code"] = x[ba_cols].idxmax(axis=1).str.replace("ba_", "", regex=False)
    return x


def forecast_origin_eligible(x):
    local = x["datetime_utc"].dt.tz_convert("America/New_York")
    return x[(local.dt.hour >= 11) & (local.dt.hour <= 22)].copy()


def split(x, start, end):
    return x[(x["datetime_utc"] >= start) & (x["datetime_utc"] <= end)].copy()


def feature_cols(x):
    cols = [c for c in x.columns if c.startswith(("residual_lag_", "demand_lag_", "resid_mean_", "resid_std_", "ba_")) and c != "ba_code"]
    cols += ["forecast", "hour", "day_of_week", "month", "is_weekend"]
    return cols


def fit_ridge(train_X, train_y, val_X, val_y):
    best = None
    for alpha in ALPHAS:
        m = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        m.fit(train_X, train_y)
        mae = mean_absolute_error(val_y, m.predict(val_X))
        if best is None or mae < best[0]:
            best = (mae, alpha, m)
    return best[2], {"alpha": best[1], "validation_mae": float(best[0])}


def fit_hgb(train_X, train_y):
    m = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("hgb", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))])
    return m.fit(train_X, train_y)


def fit_stack(pred, y):
    n = pred.shape[1]
    fun = lambda w: np.mean(np.abs(y - pred @ w))
    res = minimize(fun, np.repeat(1 / n, n), method="SLSQP", bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def metrics(y, pred, baseline=None):
    yv = np.asarray(y, dtype=float); pv = np.asarray(pred, dtype=float)
    mae = mean_absolute_error(yv, pv); rmse = math.sqrt(mean_squared_error(yv, pv))
    out = {"mae": float(mae), "rmse": float(rmse), "n": int(len(yv))}
    if baseline is not None:
        bmae = mean_absolute_error(yv, np.asarray(baseline, dtype=float))
        out["relative_mae_improvement_vs_operator"] = float((bmae - mae) / bmae) if bmae else None
    return out


def bootstrap_cluster_lift(x, pred_col, n=N_BOOT, seed=SEED):
    vals = []
    for (_, _), g in x.groupby(["ba_code", "week"]):
        b = np.mean(np.abs(g["demand"] - g["forecast"])); m = np.mean(np.abs(g["demand"] - g[pred_col]))
        if b > 0:
            vals.append((b - m) / b)
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return {"n_clusters": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed); sims = np.empty(n)
    for i in range(n):
        sims[i] = np.mean(rng.choice(a, size=len(a), replace=True))
    return {"n_clusters": int(len(a)), "mean": float(a.mean()), "ci95": [float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))]}


def ba_validation_router(val):
    decisions = {}
    for ba, g in val.groupby("ba_code"):
        g = g.copy(); g["week"] = g["datetime_utc"].dt.to_period("W").astype(str)
        b = bootstrap_cluster_lift(g, "pred_stack")
        decisions[ba] = {"use_enhanced": bool(b["ci95"][0] is not None and b["ci95"][0] > 0), "validation_weekly_bootstrap": b}
    return decisions


def main():
    outdir = Path(os.environ.get("PFSCE_OUT", "validation/pfsce_necf/results/2024_b1")); outdir.mkdir(parents=True, exist_ok=True)
    zpath = outdir / "EBA.zip"
    print("Downloading EIA bulk EBA.zip...", flush=True)
    source_meta = download(EBA_URL, zpath)
    print(f"Downloaded {source_meta['downloaded_bytes'] / 1e6:.1f} MB", flush=True)
    data, series_meta = extract_series(zpath)
    data = data[(data["datetime_utc"] >= TRAIN_START) & (data["datetime_utc"] <= TEST_END)].copy()
    data.to_csv(outdir / "eia_2020_2024_four_ba.csv", index=False)
    good, bad = quarantine(data)
    f = build_features(good); f["datetime_utc"] = pd.to_datetime(f["datetime_utc"], utc=True)
    eligible = forecast_origin_eligible(f); cols = feature_cols(eligible)
    train = split(eligible, TRAIN_START, TRAIN_END).dropna(subset=["residual"])
    val = split(eligible, VAL_START, VAL_END).dropna(subset=["residual"])
    test = split(eligible, TEST_START, TEST_END).dropna(subset=["residual"])
    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError(f"Empty split: train={len(train)} val={len(val)} test={len(test)}")
    ridge, ridge_meta = fit_ridge(train[cols], train["residual"], val[cols], val["residual"]); hgb = fit_hgb(train[cols], train["residual"])
    val["pred_ridge"] = val["forecast"] + ridge.predict(val[cols]); val["pred_hgb"] = val["forecast"] + hgb.predict(val[cols])
    val_components = np.column_stack([val["forecast"], val["pred_ridge"], val["pred_hgb"]]); weights = fit_stack(val_components, val["demand"].to_numpy()); val["pred_stack"] = val_components @ weights
    router = ba_validation_router(val)
    test["pred_ridge"] = test["forecast"] + ridge.predict(test[cols]); test["pred_hgb"] = test["forecast"] + hgb.predict(test[cols])
    test_components = np.column_stack([test["forecast"], test["pred_ridge"], test["pred_hgb"]]); test["pred_stack"] = test_components @ weights
    test["pred_routed"] = test.apply(lambda r: r["pred_stack"] if router.get(r["ba_code"], {}).get("use_enhanced", False) else r["forecast"], axis=1)
    test["week"] = test["datetime_utc"].dt.to_period("W").astype(str)
    model_map = {"operator": "forecast", "ridge": "pred_ridge", "hgb": "pred_hgb", "stack": "pred_stack", "routed_stack": "pred_routed"}
    result = {"protocol_id": PROTOCOL_ID, "protocol_tree_sha256": PROTOCOL_TREE_SHA256, "authority_label": "V1_PROVISIONAL_REVISION_SENSITIVE_CURRENT_EIA_BULK", "holdout": "2024_CONFIRMATORY", "source": source_meta, "series_metadata": series_meta, "n_raw": int(len(data)), "n_quarantined": int(len(bad)), "n_good": int(len(good)), "n_train_eligible": int(len(train)), "n_validation_eligible": int(len(val)), "n_holdout_eligible": int(len(test)), "ridge_selection": ridge_meta, "stack_weights_operator_ridge_hgb": [float(x) for x in weights], "router_frozen_from_2023": router, "models": {}, "by_ba": {}, "bootstrap": {}}
    for name, col in model_map.items():
        result["models"][name] = metrics(test["demand"], test[col], None if name == "operator" else test["forecast"])
        if name != "operator": result["bootstrap"][name] = bootstrap_cluster_lift(test, col)
    for ba, g in test.groupby("ba_code"):
        result["by_ba"][ba] = {name: metrics(g["demand"], g[col], None if name == "operator" else g["forecast"]) for name, col in model_map.items()}
    rb = result["bootstrap"]["routed_stack"]; routed_lift = result["models"]["routed_stack"].get("relative_mae_improvement_vs_operator")
    result["confirmatory_b1_verdict"] = {"routed_relative_mae_lift": routed_lift, "BA_week_ci95": rb["ci95"], "positive_cluster_robust_lift": bool(routed_lift is not None and routed_lift > 0 and rb["ci95"][0] is not None and rb["ci95"][0] > 0), "note": "Current EIA bulk is revision-sensitive; positive result remains V1 provisional until vintage sensitivity is quantified."}
    (outdir / "B1_2024_confirmatory.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    test.to_parquet(outdir / "B1_2024_predictions.parquet", index=False); bad.to_csv(outdir / "quarantined_rows.csv", index=False)
    manifest = {"protocol_id": PROTOCOL_ID, "protocol_tree_sha256": PROTOCOL_TREE_SHA256, "result_sha256": sha256_file(outdir / "B1_2024_confirmatory.json"), "predictions_sha256": sha256_file(outdir / "B1_2024_predictions.parquet"), "source_zip_sha256": source_meta["sha256"], "python": sys.version}
    (outdir / "RESULT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["confirmatory_b1_verdict"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
