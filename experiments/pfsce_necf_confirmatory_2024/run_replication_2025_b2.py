from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_b1 as b1
import run_b2 as b2
import run_replication_2025_b1 as r25

REPLICATION_ID = "NECF-001-WP2-REPLICATION-2025-v1"
SOURCE_ADAPTER = "POINT_IN_TIME_ELIGIBLE_DERIVED_HRRR_MIRROR"
EXPECTED_PROMOTED = {"CISO": True, "ERCO": False, "PJM": True, "ISNE": False}
WEATHER_COLS = b2.WEATHER_COLS
SEED = 20260819
BOOT = 5000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def load_combined_eia() -> tuple[pd.DataFrame, dict]:
    historical = b1.load_data()
    historical = historical[historical.datetime_utc <= pd.Timestamp("2024-12-31T23:59:59Z")].copy()
    target, prov = r25.load_2025_target()
    combined = pd.concat([historical, target], ignore_index=True)
    combined = combined.sort_values(["ba", "datetime_utc"]).drop_duplicates(["ba", "datetime_utc"], keep="last")
    return combined, prov


def load_weather(weather_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted(weather_dir.rglob("weather_*.csv"))
    if not files:
        raise RuntimeError(f"No weather CSV shards found under {weather_dir}")
    frames = []
    for p in files:
        q = pd.read_csv(p)
        if {"datetime_utc", "ba_code"}.issubset(q.columns):
            frames.append(q)
    if not frames:
        raise RuntimeError("Weather shards contained no usable datetime_utc/ba_code data")
    w = pd.concat(frames, ignore_index=True)
    w["datetime_utc"] = pd.to_datetime(w.datetime_utc, utc=True)
    w["ba_code"] = w.ba_code.astype(str)
    w = w.sort_values(["datetime_utc", "ba_code"]).drop_duplicates(["datetime_utc", "ba_code"], keep="last")
    manifests = []
    for p in sorted(weather_dir.rglob("weather_*_manifest.json")):
        try:
            manifests.append(json.loads(p.read_text()))
        except Exception:
            manifests.append({"path": str(p), "parse_error": True})
    return w, manifests


def fit_frozen_system(
    f: pd.DataFrame,
    features: list[str],
    frozen_weights: np.ndarray,
    expected_router: dict[str, bool],
    expected_alpha: float,
) -> tuple[dict, pd.DataFrame]:
    train = b1.split(f, "2020-01-01T00:00:00Z", "2022-12-31T23:00:00Z").dropna(subset=["demand", "forecast", "residual"])
    val = b1.split(f, "2023-01-01T00:00:00Z", "2023-12-31T23:00:00Z").dropna(subset=["demand", "forecast", "residual"])
    rep = b1.split(f, "2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z").dropna(subset=["demand", "forecast", "residual"])
    if min(len(train), len(val), len(rep)) == 0:
        raise RuntimeError(f"Empty frozen cohort train={len(train)} val={len(val)} rep={len(rep)}")

    ridge, ridge_meta = b1.ridge_fit(train[features], train.residual, val[features], val.residual)
    if float(ridge_meta["alpha"]) != float(expected_alpha):
        raise RuntimeError(f"Frozen alpha not reproduced: {ridge_meta}")
    hgb = b1.hgb_fit(train[features], train.residual)

    vr = val.forecast.to_numpy() + ridge.predict(val[features])
    vh = val.forecast.to_numpy() + hgb.predict(val[features])
    val = val.copy()
    val["pred_stack"] = np.column_stack([val.forecast.to_numpy(), vr, vh]) @ frozen_weights
    weekly_val = b1.weekly_lifts(val, "pred_stack")
    reproduced_router = {}
    for ba in b1.BAS:
        ci = b1.boot([v for bb, _, v in weekly_val if bb == ba])
        promoted = bool(ci["lower"] is not None and ci["lower"] > 0)
        reproduced_router[ba] = {"promoted": promoted, "validation_BA_week_relative_lift_bootstrap": ci}
        if promoted != expected_router[ba]:
            raise RuntimeError(f"Frozen B2 router not reproduced for {ba}: {promoted} != {expected_router[ba]} {ci}")

    rr = rep.forecast.to_numpy() + ridge.predict(rep[features])
    rh = rep.forecast.to_numpy() + hgb.predict(rep[features])
    rs = np.column_stack([rep.forecast.to_numpy(), rr, rh]) @ frozen_weights
    rep = rep.copy()
    rep["pred_ridge"] = rr
    rep["pred_hgb"] = rh
    rep["pred_stack"] = rs
    rep["router_promoted"] = rep.ba_code.map(expected_router)
    rep["pred_routed"] = np.where(rep.router_promoted, rep.pred_stack, rep.forecast)

    models = {}
    for name, col in [("operator", "forecast"), ("ridge", "pred_ridge"), ("hgb", "pred_hgb"), ("stack", "pred_stack"), ("routed", "pred_routed")]:
        models[name] = b1.metrics(rep.demand, rep[col], rep.forecast if name != "operator" else None)
    return {
        "ridge": ridge_meta,
        "stack_weights": frozen_weights.tolist(),
        "router_reproduction_2023": reproduced_router,
        "models": models,
        "n_train": int(len(train)),
        "n_validation": int(len(val)),
        "n_replication": int(len(rep)),
    }, rep


def paired_weekly_lift(df: pd.DataFrame, baseline_col: str, challenger_col: str) -> pd.DataFrame:
    q = df.copy()
    q["week"] = q.datetime_utc.dt.tz_localize(None).dt.to_period("W").astype(str)
    rows = []
    for (ba, week), g in q.groupby(["ba_code", "week"]):
        bm = float(np.mean(np.abs(g.demand - g[baseline_col])))
        cm = float(np.mean(np.abs(g.demand - g[challenger_col])))
        if bm > 0:
            rows.append({"ba_code": str(ba), "week": week, "relative_lift": (bm - cm) / bm, "baseline_mae": bm, "challenger_mae": cm})
    return pd.DataFrame(rows)


def main(weather_dir: Path, out_dir: Path) -> None:
    frozen_b1 = json.loads(Path("experiments/pfsce_necf_confirmatory_2024/results/B1_2024.json").read_text())
    frozen_b2 = json.loads(Path("experiments/pfsce_necf_confirmatory_2024/results/B2_2024.json").read_text())
    weights_b1 = np.asarray(frozen_b1["stack_weights"], float)
    weights_b2 = np.asarray(frozen_b2["B2"]["stack_weights"], float)
    alpha_b1 = float(frozen_b1["ridge"]["alpha"])
    alpha_b2 = float(frozen_b2["B2"]["ridge"]["alpha"])

    raw, source_2025 = load_combined_eia()
    good, bad = b1.quarantine(raw)
    base = b1.target_filter(b1.features(good))
    F1 = b1.cols(base)

    w, weather_manifests = load_weather(weather_dir)
    fw = base.merge(w, on=["datetime_utc", "ba_code"], how="left")
    F2 = F1 + WEATHER_COLS

    b1res, b1rep = fit_frozen_system(base, F1, weights_b1, EXPECTED_PROMOTED, alpha_b1)
    b2res, b2rep = fit_frozen_system(fw, F2, weights_b2, EXPECTED_PROMOTED, alpha_b2)

    keys = ["datetime_utc", "ba_code"]
    comp = b1rep[keys + ["demand", "forecast", "pred_routed"]].rename(columns={"pred_routed": "pred_b1"}).merge(
        b2rep[keys + ["pred_routed", "router_promoted"]].rename(columns={"pred_routed": "pred_b2", "router_promoted": "b2_router_promoted"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(comp) != len(b1rep) or len(comp) != len(b2rep):
        raise RuntimeError(f"B1/B2 replication alignment mismatch {len(comp)} {len(b1rep)} {len(b2rep)}")

    mae1 = float(np.mean(np.abs(comp.demand - comp.pred_b1)))
    mae2 = float(np.mean(np.abs(comp.demand - comp.pred_b2)))
    aggregate = (mae1 - mae2) / mae1
    weekly = paired_weekly_lift(comp, "pred_b1", "pred_b2")
    ci = b1.boot(weekly.relative_lift.to_numpy())

    by_ba = {}
    positive_median = 0
    for ba, g in weekly.groupby("ba_code"):
        med = float(g.relative_lift.median())
        if med > 0:
            positive_median += 1
        cg = comp[comp.ba_code == ba]
        bm = float(np.mean(np.abs(cg.demand - cg.pred_b1)))
        cm = float(np.mean(np.abs(cg.demand - cg.pred_b2)))
        by_ba[str(ba)] = {
            "median_weekly_improvement_vs_B1": med,
            "mean_weekly_improvement_vs_B1": float(g.relative_lift.mean()),
            "BA_week_bootstrap": b1.boot(g.relative_lift.to_numpy()),
            "B1_mae": bm,
            "B2_mae": cm,
            "relative_mae_improvement_vs_B1": (bm - cm) / bm,
            "B2_router_promoted": bool(cg.b2_router_promoted.iloc[0]),
        }

    unrouted_degradations = [max(0.0, -v["relative_mae_improvement_vs_B1"]) for v in by_ba.values() if not v["B2_router_promoted"]]
    max_unrouted = max(unrouted_degradations) if unrouted_degradations else 0.0
    conditions = {
        "aggregate_incremental_lift_ge_2pct": aggregate >= 0.02,
        "paired_BA_week_bootstrap_lower_gt_zero": ci["lower"] is not None and ci["lower"] > 0,
        "at_least_3_of_4_BAs_positive_median_weekly": positive_median >= 3,
        "max_unrouted_BA_degradation_le_5pct": max_unrouted <= 0.05,
    }
    descriptive_gate_pass = all(conditions.values())
    interpretation = "B2_V1_2025_DESCRIPTIVE_GATE_PASS_JUSTIFIES_NEW_PREREGISTERED_B2_V2" if descriptive_gate_pass else "B2_V1_REPLICATION_FAIL_STRENGTHENS_NON_PROMOTION"

    coverage = {}
    for label, a, z in [
        ("train", "2020-01-01T00:00:00Z", "2022-12-31T23:00:00Z"),
        ("validation", "2023-01-01T00:00:00Z", "2023-12-31T23:00:00Z"),
        ("replication", "2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z"),
    ]:
        s = b1.split(fw, a, z)
        complete = s[WEATHER_COLS].notna().all(axis=1)
        coverage[label] = {"rows": int(len(s)), "weather_complete_rows": int(complete.sum()), "coverage": float(complete.mean()) if len(s) else None}

    result = {
        "replication_id": REPLICATION_ID,
        "status": "FROZEN_B2_2025_REPLICATION_RESULT",
        "source_adapter": SOURCE_ADAPTER,
        "EIA_source_2025": source_2025,
        "frozen_2024_B1_result_sha256": sha256_file(Path("experiments/pfsce_necf_confirmatory_2024/results/B1_2024.json")),
        "frozen_2024_B2_result_sha256": sha256_file(Path("experiments/pfsce_necf_confirmatory_2024/results/B2_2024.json")),
        "B1_2025_reproduction": b1res,
        "B2_2025": b2res,
        "weather_coverage": coverage,
        "incremental_replication_gate_vs_B1": {
            "B1_routed_mae": mae1,
            "B2_routed_mae": mae2,
            "aggregate_relative_mae_improvement_vs_B1": aggregate,
            "paired_BA_week_bootstrap": ci,
            "BAs_positive_median_weekly_improvement": positive_median,
            "max_unrouted_BA_degradation": max_unrouted,
            "conditions": conditions,
            "descriptive_gate_pass": descriptive_gate_pass,
            "interpretation": interpretation,
            "note": "The 2024 B2 verdict remains FAIL regardless of this replication result. A 2025 pass can only justify a new preregistered B2-v2 experiment.",
        },
        "by_ba_incremental_vs_B1": by_ba,
        "n_quarantined_combined": int(len(bad)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / "B2_2025_REPLICATION.json"
    rp.write_text(json.dumps(result, indent=2, sort_keys=True))
    comp.to_csv(out_dir / "B2_2025_comparison_predictions.csv.gz", index=False, compression="gzip")
    weather_prov = {
        "adapter": SOURCE_ADAPTER,
        "manifests": weather_manifests,
        "weather_rows_combined": int(len(w)),
        "weather_combined_sha256": hashlib.sha256(w.to_csv(index=False).encode()).hexdigest(),
    }
    (out_dir / "B2_2025_WEATHER_PROVENANCE.json").write_text(json.dumps(weather_prov, indent=2, sort_keys=True))

    lines = [
        "# B2 2025 Replication Verdict",
        "",
        f"Replication: `{REPLICATION_ID}`",
        "",
        f"B1 routed MAE: **{mae1:.3f} MW**",
        f"B2 routed MAE: **{mae2:.3f} MW**",
        f"Incremental aggregate MAE improvement vs frozen B1: **{aggregate*100:.3f}%**",
        f"Paired BA-week bootstrap 95% CI: **[{ci['lower']*100:.3f}%, {ci['upper']*100:.3f}%]**",
        f"BAs with positive median weekly improvement: **{positive_median}/4**",
        f"Maximum degradation among B2-unrouted BAs: **{max_unrouted*100:.3f}%**",
        "",
        f"**Descriptive 2025 frozen gate: {'PASS' if descriptive_gate_pass else 'FAIL'}.**",
        f"**Interpretation: {interpretation}.**",
        "",
        "The 2024 B2 confirmatory verdict remains FAIL. This replication cannot retroactively promote B2-v1.",
        "",
        *[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in conditions.items()],
    ]
    (out_dir / "B2_2025_REPLICATION_VERDICT.md").write_text("\n".join(lines) + "\n")
    checksums = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file()}
    (out_dir / "B2_2025_SHA256.json").write_text(json.dumps(checksums, indent=2, sort_keys=True))

    print("PFSCE_B2_2025_REPLICATION_BEGIN")
    print(json.dumps(result, indent=2))
    print("PFSCE_B2_2025_REPLICATION_END")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("experiment_output/replication_2025_b2"), type=Path)
    args = ap.parse_args()
    main(args.weather_dir, args.out_dir)
