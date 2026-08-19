from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROTOCOL_ID = "PFSCE-V8.2-FLUSIGHT-RESIDUAL-V1"
SEED = 20260820
BOOT = 5000
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
HORIZONS = [1, 2, 3]
TARGET = "wk inc flu hosp"
TRAIN = (pd.Timestamp("2023-10-01"), pd.Timestamp("2024-06-30"))
VALID = (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-06-30"))
HOLD = (pd.Timestamp("2025-10-01"), pd.Timestamp("2026-06-30"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def pinball(y: np.ndarray, pred: np.ndarray, q: np.ndarray) -> np.ndarray:
    e = y - pred
    return np.maximum(q * e, (q - 1.0) * e)


def load_ensemble(repo: Path) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted((repo / "model-output" / "FluSight-ensemble").glob("*.csv"))
    frames = []
    provenance = []
    for p in files:
        try:
            d = pd.read_csv(p, dtype={"location": str, "output_type_id": str})
        except Exception as e:
            provenance.append({"file": str(p.relative_to(repo)), "sha256": sha256(p), "read_error": repr(e)})
            continue
        needed = {"reference_date", "location", "horizon", "target", "target_end_date", "output_type", "output_type_id", "value"}
        if not needed.issubset(d.columns):
            provenance.append({"file": str(p.relative_to(repo)), "sha256": sha256(p), "skipped": "schema"})
            continue
        d["reference_date"] = pd.to_datetime(d["reference_date"], errors="coerce")
        d["target_end_date"] = pd.to_datetime(d["target_end_date"], errors="coerce")
        d["horizon"] = pd.to_numeric(d["horizon"], errors="coerce")
        d["q"] = pd.to_numeric(d["output_type_id"], errors="coerce")
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d[(d.target == TARGET) & (d.output_type == "quantile") & (d.horizon.isin(HORIZONS)) & d.q.notna() & d.value.notna()].copy()
        if not d.empty:
            frames.append(d[["reference_date", "location", "horizon", "target_end_date", "q", "value"]])
            provenance.append({"file": str(p.relative_to(repo)), "sha256": sha256(p), "rows_selected": int(len(d)), "reference_date": str(d.reference_date.iloc[0].date())})
    if not frames:
        raise RuntimeError("No FluSight ensemble quantile forecasts found")
    x = pd.concat(frames, ignore_index=True)
    x["location"] = x.location.astype(str).str.zfill(2).where(x.location.astype(str) != "US", "US")
    x = x.drop_duplicates(["reference_date", "location", "horizon", "target_end_date", "q"], keep="last")
    return x, provenance


def load_truth(repo: Path) -> tuple[pd.DataFrame, dict]:
    p = repo / "target-data" / "target-hospital-admissions.csv"
    d = pd.read_csv(p, dtype={"location": str})
    required = {"date", "location", "value"}
    if not required.issubset(d.columns):
        raise RuntimeError(f"Truth schema mismatch: {d.columns.tolist()}")
    d["target_end_date"] = pd.to_datetime(d["date"], errors="coerce")
    d["truth"] = pd.to_numeric(d["value"], errors="coerce")
    d["location"] = d.location.astype(str).str.zfill(2).where(d.location.astype(str) != "US", "US")
    d = d.dropna(subset=["target_end_date", "truth"]).sort_values(["target_end_date", "location"])
    d = d.drop_duplicates(["target_end_date", "location"], keep="last")
    return d[["target_end_date", "location", "truth"]], {"file": str(p.relative_to(repo)), "sha256": sha256(p), "rows": int(len(d))}


def cohort(df: pd.DataFrame, bounds: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    a, b = bounds
    return df[(df.reference_date >= a) & (df.reference_date <= b)].copy()


def cases_from_quantiles(qf: pd.DataFrame) -> pd.DataFrame:
    med = qf[np.isclose(qf.q, 0.5)][["reference_date", "location", "horizon", "target_end_date", "value", "truth"]].copy()
    med = med.rename(columns={"value": "baseline_median"})
    med = med.drop_duplicates(["reference_date", "location", "horizon", "target_end_date"])
    med["residual"] = med.truth - med.baseline_median
    med["log_baseline"] = np.log1p(np.clip(med.baseline_median, 0, None))
    doy = med.reference_date.dt.dayofyear.astype(float)
    med["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    med["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    med["h_log"] = med.horizon.astype(float) * med.log_baseline
    return med


def ridge_pipeline(alpha: float) -> Pipeline:
    cats = ["location", "horizon"]
    nums = ["log_baseline", "sin_year", "cos_year", "h_log"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", StandardScaler(), nums),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def horizon_bias_fit(cases: pd.DataFrame) -> dict[int, float]:
    return {int(h): float(g.residual.mean()) for h, g in cases.groupby("horizon")}


def correction_for_cases(cases: pd.DataFrame, kind: str, model) -> np.ndarray:
    if kind == "NO_CORRECTION":
        return np.zeros(len(cases), dtype=float)
    if kind == "HORIZON_BIAS":
        return cases.horizon.map(lambda h: model.get(int(h), 0.0)).astype(float).to_numpy()
    if kind == "RIDGE_RESIDUAL":
        return model.predict(cases)
    raise ValueError(kind)


def attach_correction(qf: pd.DataFrame, cases: pd.DataFrame, corr: np.ndarray, col="correction") -> pd.DataFrame:
    key = ["reference_date", "location", "horizon", "target_end_date"]
    c = cases[key].copy()
    c[col] = corr
    out = qf.merge(c, on=key, how="inner", validate="many_to_one")
    return out


def score_quantiles(d: pd.DataFrame, pred_col: str) -> float:
    return float(pinball(d.truth.to_numpy(float), d[pred_col].to_numpy(float), d.q.to_numpy(float)).mean())


def score_median_mae(d: pd.DataFrame, pred_col: str) -> float:
    m = d[np.isclose(d.q, 0.5)].copy()
    return float(np.mean(np.abs(m.truth - m[pred_col])))


def interval_coverage(d: pd.DataFrame, pred_col: str) -> dict[str, float | None]:
    keys = ["reference_date", "location", "horizon", "target_end_date", "truth"]
    piv = d.pivot_table(index=keys, columns="q", values=pred_col, aggfunc="last").reset_index()
    out = {}
    for nominal, lo, hi in [(0.50, 0.25, 0.75), (0.80, 0.10, 0.90), (0.90, 0.05, 0.95), (0.95, 0.025, 0.975)]:
        if lo in piv.columns and hi in piv.columns:
            ok = (piv.truth >= piv[lo]) & (piv.truth <= piv[hi])
            out[f"{int(nominal*100)}"] = float(ok.mean())
        else:
            out[f"{int(nominal*100)}"] = None
    return out


def mean_abs_coverage_error(cov: dict[str, float | None]) -> float | None:
    vals = []
    for k, v in cov.items():
        if v is not None:
            vals.append(abs(v - int(k) / 100.0))
    return float(np.mean(vals)) if vals else None


def weekly_relative_lifts(d: pd.DataFrame, challenger_col: str) -> pd.DataFrame:
    rows = []
    for rd, g in d.groupby("reference_date"):
        b = score_quantiles(g, "value")
        c = score_quantiles(g, challenger_col)
        if b > 0:
            rows.append({"reference_date": str(pd.Timestamp(rd).date()), "baseline": b, "challenger": c, "relative_lift": (b-c)/b})
    return pd.DataFrame(rows)


def bootstrap(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return {"mean": None, "median": None, "lower": None, "upper": None, "n_clusters": 0}
    rng = np.random.default_rng(SEED)
    sims = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(BOOT)])
    return {"mean": float(vals.mean()), "median": float(np.median(vals)), "lower": float(np.quantile(sims, 0.025)), "upper": float(np.quantile(sims, 0.975)), "n_clusters": int(len(vals))}


def candidate_validation(train_cases, val_cases, val_q):
    candidates = []

    # baseline
    v0 = val_q.copy(); v0["pred"] = v0.value
    candidates.append({"kind": "NO_CORRECTION", "model": None, "pinball": score_quantiles(v0, "pred"), "details": {}})

    # horizon bias
    hb = horizon_bias_fit(train_cases)
    hc = correction_for_cases(val_cases, "HORIZON_BIAS", hb)
    vh = attach_correction(val_q, val_cases, hc); vh["pred"] = vh.value + vh.correction
    candidates.append({"kind": "HORIZON_BIAS", "model": hb, "pinball": score_quantiles(vh, "pred"), "details": {"bias": hb}})

    # ridge alpha selected on validation proper score
    ridge_rows = []
    ridge_models = {}
    for a in ALPHAS:
        m = ridge_pipeline(a); m.fit(train_cases, train_cases.residual)
        corr = correction_for_cases(val_cases, "RIDGE_RESIDUAL", m)
        vr = attach_correction(val_q, val_cases, corr); vr["pred"] = vr.value + vr.correction
        s = score_quantiles(vr, "pred")
        ridge_rows.append({"alpha": a, "pinball": s})
        ridge_models[a] = m
    best = min(ridge_rows, key=lambda x: (x["pinball"], x["alpha"]))
    candidates.append({"kind": "RIDGE_RESIDUAL", "model": ridge_models[best["alpha"]], "pinball": best["pinball"], "details": {"alpha": best["alpha"], "all": ridge_rows}})

    # 0.25% tie -> simpler in predefined order
    order = {"NO_CORRECTION": 0, "HORIZON_BIAS": 1, "RIDGE_RESIDUAL": 2}
    best_loss = min(c["pinball"] for c in candidates)
    eligible = [c for c in candidates if c["pinball"] <= best_loss * 1.0025]
    chosen = min(eligible, key=lambda c: order[c["kind"]])
    return candidates, chosen


def fit_selected_on_train(train_cases: pd.DataFrame, chosen: dict):
    kind = chosen["kind"]
    if kind == "NO_CORRECTION": return None
    if kind == "HORIZON_BIAS": return horizon_bias_fit(train_cases)
    alpha = float(chosen["details"]["alpha"])
    m = ridge_pipeline(alpha); m.fit(train_cases, train_cases.residual); return m


def main(repo: Path, outdir: Path):
    qf, forecast_prov = load_ensemble(repo)
    truth, truth_prov = load_truth(repo)
    qf = qf.merge(truth, on=["target_end_date", "location"], how="left")
    total_forecast_rows = len(qf)
    missing_truth_rows = int(qf.truth.isna().sum())
    qf = qf.dropna(subset=["truth"]).copy()

    train_q, val_q, hold_q = cohort(qf, TRAIN), cohort(qf, VALID), cohort(qf, HOLD)
    train_cases, val_cases, hold_cases = cases_from_quantiles(train_q), cases_from_quantiles(val_q), cases_from_quantiles(hold_q)
    if min(len(train_cases), len(val_cases), len(hold_cases)) == 0:
        raise RuntimeError(f"Empty cohort: train={len(train_cases)} val={len(val_cases)} hold={len(hold_cases)}")

    candidates, chosen = candidate_validation(train_cases, val_cases, val_q)
    fitted = fit_selected_on_train(train_cases, chosen)

    # validation router by horizon
    val_corr = correction_for_cases(val_cases, chosen["kind"], fitted)
    v = attach_correction(val_q, val_cases, val_corr); v["challenger"] = v.value + v.correction
    router = {}
    for h in HORIZONS:
        vh = v[v.horizon == h]
        wk = weekly_relative_lifts(vh, "challenger")
        ci = bootstrap(wk.relative_lift.to_numpy())
        router[str(h)] = {"promoted": bool(ci["lower"] is not None and ci["lower"] > 0), "validation_reference_week_bootstrap": ci, "validation_relative_pinball_lift": (score_quantiles(vh, "value")-score_quantiles(vh, "challenger"))/score_quantiles(vh, "value")}

    # holdout frozen policy
    hc = correction_for_cases(hold_cases, chosen["kind"], fitted)
    h = attach_correction(hold_q, hold_cases, hc); h["challenger"] = h.value + h.correction
    h["router_promoted"] = h.horizon.astype(str).map(lambda x: router[x]["promoted"])
    h["routed"] = np.where(h.router_promoted, h.challenger, h.value)

    base_pin = score_quantiles(h, "value"); routed_pin = score_quantiles(h, "routed")
    agg_lift = (base_pin-routed_pin)/base_pin
    wk = weekly_relative_lifts(h, "routed"); ci = bootstrap(wk.relative_lift.to_numpy())
    base_mae = score_median_mae(h, "value"); routed_mae = score_median_mae(h, "routed")
    mae_change = (routed_mae-base_mae)/base_mae
    base_cov = interval_coverage(h, "value"); routed_cov = interval_coverage(h, "routed")
    base_cerr = mean_abs_coverage_error(base_cov); routed_cerr = mean_abs_coverage_error(routed_cov)
    cov_worsen = None if base_cerr is None or routed_cerr is None else routed_cerr-base_cerr

    by_h = {}
    promoted_nonnegative = True
    for hh, g in h.groupby("horizon"):
        bp=score_quantiles(g,"value"); rp=score_quantiles(g,"routed"); lift=(bp-rp)/bp
        by_h[str(int(hh))] = {"promoted": bool(router[str(int(hh))]["promoted"]), "baseline_pinball":bp,"routed_pinball":rp,"relative_pinball_lift":lift,"baseline_median_mae":score_median_mae(g,"value"),"routed_median_mae":score_median_mae(g,"routed")}
        if router[str(int(hh))]["promoted"] and lift < 0: promoted_nonnegative=False

    conditions = {
        "aggregate_pinball_lift_ge_2pct": agg_lift >= 0.02,
        "reference_week_bootstrap_lower_gt_zero": ci["lower"] is not None and ci["lower"] > 0,
        "every_promoted_horizon_nonnegative": bool(promoted_nonnegative),
        "median_mae_degradation_le_1pct": mae_change <= 0.01,
        "coverage_error_worsening_le_0_02": cov_worsen is not None and cov_worsen <= 0.02,
    }
    if all(conditions.values()): verdict="CROSS_DOMAIN_PASS"
    elif agg_lift > 0 and ci["lower"] is not None and ci["lower"] > 0: verdict="DIRECTIONAL_TRANSFER"
    else: verdict="NO_TRANSFER"

    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_HOLDOUT_RESULT",
        "source_repo": "cdcepi/FluSight-forecast-hub",
        "source_commit": (repo / ".git" / "HEAD").read_text().strip() if (repo/".git"/"HEAD").exists() else None,
        "source_truth": truth_prov,
        "forecast_files_count": len(forecast_prov),
        "cohorts": {"train_cases":len(train_cases),"validation_cases":len(val_cases),"holdout_cases":len(hold_cases),"train_reference_weeks":int(train_cases.reference_date.nunique()),"validation_reference_weeks":int(val_cases.reference_date.nunique()),"holdout_reference_weeks":int(hold_cases.reference_date.nunique())},
        "denominator": {"forecast_rows_before_truth_merge":total_forecast_rows,"missing_truth_rows":missing_truth_rows,"scored_holdout_quantile_rows":len(h),"holdout_locations":int(h.location.nunique())},
        "validation_candidates": [{"kind":c["kind"],"pinball":c["pinball"],"details":c["details"]} for c in candidates],
        "selected_challenger": {"kind":chosen["kind"],"pinball":chosen["pinball"],"details":chosen["details"]},
        "router": router,
        "holdout": {"baseline_pinball":base_pin,"routed_pinball":routed_pin,"relative_pinball_lift":agg_lift,"reference_week_bootstrap":ci,"baseline_median_mae":base_mae,"routed_median_mae":routed_mae,"relative_median_mae_change":mae_change,"baseline_coverage":base_cov,"routed_coverage":routed_cov,"baseline_mean_abs_coverage_error":base_cerr,"routed_mean_abs_coverage_error":routed_cerr,"coverage_error_worsening":cov_worsen,"by_horizon":by_h},
        "promotion_gate": {"conditions":conditions,"verdict":verdict},
        "authority": "RETROSPECTIVE_CROSS_DOMAIN_V1_DIAGNOSTIC_NOT_PROSPECTIVE",
    }

    outdir.mkdir(parents=True, exist_ok=True)
    rp=outdir/"FLUSIGHT_RESIDUAL_V1_RESULT.json"; rp.write_text(json.dumps(result, indent=2, sort_keys=True))
    h.to_csv(outdir/"FLUSIGHT_RESIDUAL_V1_HOLDOUT_QUANTILES.csv.gz", index=False, compression="gzip")
    (outdir/"FLUSIGHT_RESIDUAL_V1_SOURCE_PROVENANCE.json").write_text(json.dumps({"truth":truth_prov,"forecast_files":forecast_prov},indent=2,sort_keys=True))
    lines=["# FluSight Cross-Domain Residual Validation", "", f"Protocol: `{PROTOCOL_ID}`", f"**Verdict: {verdict}.**", "", f"Selected challenger: **{chosen['kind']}**", f"Holdout baseline mean pinball: **{base_pin:.6f}**", f"Holdout routed mean pinball: **{routed_pin:.6f}**", f"Relative pinball improvement: **{agg_lift*100:.3f}%**", f"Reference-week bootstrap 95% CI: **[{ci['lower']*100:.3f}%, {ci['upper']*100:.3f}%]**", f"Baseline median MAE: **{base_mae:.3f}**", f"Routed median MAE: **{routed_mae:.3f}**", "", "## Gate", *[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k,v in conditions.items()], "", "This is retrospective cross-domain evidence. It cannot establish prospective F2 authority or general PFSCE superiority."]
    (outdir/"FLUSIGHT_RESIDUAL_V1_VERDICT.md").write_text("\n".join(lines)+"\n")
    checks={p.name:sha256(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    (outdir/"SHA256.json").write_text(json.dumps(checks,indent=2,sort_keys=True))
    print("PFSCE_FLUSIGHT_RESULT_BEGIN")
    print(json.dumps(result,indent=2))
    print("PFSCE_FLUSIGHT_RESULT_END")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--flusight-repo",type=Path,required=True); ap.add_argument("--out-dir",type=Path,default=Path("experiment_output/flusight_v1")); args=ap.parse_args(); main(args.flusight_repo,args.out_dir)
