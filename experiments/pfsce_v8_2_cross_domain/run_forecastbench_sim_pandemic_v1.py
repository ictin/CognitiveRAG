from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pandemic_world.scenarios import sample_scenarios, build_corpus

PROTOCOL_ID = "PFSCE-V8.2-FBSIM-PANDEMIC-V1"
SCENARIO_SEED = 20260820
BOOT_SEED = 20260820
N_SCENARIOS = 120
BOOT = 5000
CS = [0.1, 1.0, 10.0]
PCLIP = (0.005, 0.995)
SNAPSHOT_DAY = 20

TRAIN_IDS = set(range(0, 60))
VALID_IDS = set(range(60, 90))
HOLDOUT_IDS = set(range(90, 120))

B1_FEATURES = [
    "kind_conditional", "horizon_60",
    "log_cases_R20", "log_cases_S20",
    "log_active_R20", "log_active_S20",
    "log_deaths_R20", "log_deaths_S20",
    "case_diff20", "log_case_ratio20",
    "active_diff20", "log_active_ratio20",
]

B2_EXTRA = [
    "inc_R_0_5", "inc_R_5_10", "inc_R_10_15", "inc_R_15_20",
    "inc_S_0_5", "inc_S_5_10", "inc_S_10_15", "inc_S_15_20",
    "log_growth_R_5", "log_growth_S_5",
    "log_growth_R_10", "log_growth_S_10",
    "recent_inc_diff", "log_recent_inc_ratio",
    "active_case_frac_R", "active_case_frac_S",
    "accel_R", "accel_S",
]

B3_EXTRA = [
    "intervention_present", "vax_efficacy", "vax_coverage", "vax_day",
    "efficacy_x_coverage", "days_snapshot_to_vax", "vax_already_started",
    "intervention_exposure_days", "effcov_x_exposure_days",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), PCLIP[0], PCLIP[1])


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = clip_prob(p)
    return float(np.mean((p - y) ** 2))


def safe_log_loss(y, p) -> float:
    return float(log_loss(np.asarray(y, dtype=int), clip_prob(p), labels=[0, 1]))


def ece10(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = clip_prob(p)
    edges = np.linspace(0, 1, 11)
    total = len(y)
    ece = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        if i == 9:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def calibration_intercept_slope(y, p):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    if len(np.unique(y)) < 2:
        return {"intercept": None, "slope": None}
    z = np.log(p / (1 - p)).reshape(-1, 1)
    # Very weak regularization approximates ordinary logistic calibration fit.
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    m.fit(z, y)
    return {"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0, 0])}


def parse_region(context: str, name: str) -> dict:
    # Example: Riverton: cumulative cases d0:25 -> d5:... -> d20:...
    pat = rf"{re.escape(name)}: cumulative cases ([^\n]+)\n\s*\(day 20: ([0-9.]+) active, ([0-9.]+) deaths\)"
    m = re.search(pat, context)
    if not m:
        raise ValueError(f"Could not parse {name} from context:\n{context}")
    trajectory = {}
    for d, v in re.findall(r"d(\d+):([0-9.]+)", m.group(1)):
        trajectory[int(d)] = float(v)
    required = [0, 5, 10, 15, 20]
    if any(d not in trajectory for d in required):
        raise ValueError(f"Missing {name} trajectory points: {trajectory}")
    return {
        "cases": {d: trajectory[d] for d in required},
        "active": float(m.group(2)),
        "deaths": float(m.group(3)),
    }


def parse_intervention(context: str) -> dict:
    m = re.search(
        r"PLANNED INTERVENTION: Riverton will run a vaccination campaign on day (\d+) "
        r"\((\d+)% efficacy, (\d+)% coverage\)",
        context,
    )
    if not m:
        return {"present": 0.0, "day": 0.0, "efficacy": 0.0, "coverage": 0.0}
    return {
        "present": 1.0,
        "day": float(m.group(1)),
        "efficacy": float(m.group(2)) / 100.0,
        "coverage": float(m.group(3)) / 100.0,
    }


def feature_row(item: dict) -> dict:
    context = item["context"]
    r = parse_region(context, "Riverton")
    s = parse_region(context, "Southbay")
    vax = parse_intervention(context)
    horizon = int(item["horizon"])
    conditional = 1.0 if item["kind"] == "conditional" else 0.0

    rc, sc = r["cases"], s["cases"]
    rinc = [rc[5] - rc[0], rc[10] - rc[5], rc[15] - rc[10], rc[20] - rc[15]]
    sinc = [sc[5] - sc[0], sc[10] - sc[5], sc[15] - sc[10], sc[20] - sc[15]]

    effcov = vax["efficacy"] * vax["coverage"]
    days_to_vax = (vax["day"] - SNAPSHOT_DAY) if vax["present"] else 0.0
    already = 1.0 if (vax["present"] and vax["day"] <= SNAPSHOT_DAY) else 0.0
    exposure = max(0.0, horizon - vax["day"]) if vax["present"] else 0.0

    row = {
        "scenario_id": int(item["scenario_id"]),
        "question_id": item["question_id"],
        "kind": item["kind"],
        "horizon": horizon,
        "y": int(bool(item["ground_truth"])),
        "kind_conditional": conditional,
        "horizon_60": 1.0 if horizon == 60 else 0.0,
        "log_cases_R20": math.log1p(rc[20]),
        "log_cases_S20": math.log1p(sc[20]),
        "log_active_R20": math.log1p(r["active"]),
        "log_active_S20": math.log1p(s["active"]),
        "log_deaths_R20": math.log1p(r["deaths"]),
        "log_deaths_S20": math.log1p(s["deaths"]),
        "case_diff20": rc[20] - sc[20],
        "log_case_ratio20": math.log((rc[20] + 1.0) / (sc[20] + 1.0)),
        "active_diff20": r["active"] - s["active"],
        "log_active_ratio20": math.log((r["active"] + 1.0) / (s["active"] + 1.0)),
        "inc_R_0_5": rinc[0], "inc_R_5_10": rinc[1], "inc_R_10_15": rinc[2], "inc_R_15_20": rinc[3],
        "inc_S_0_5": sinc[0], "inc_S_5_10": sinc[1], "inc_S_10_15": sinc[2], "inc_S_15_20": sinc[3],
        "log_growth_R_5": math.log((rc[20] + 1.0) / (rc[15] + 1.0)),
        "log_growth_S_5": math.log((sc[20] + 1.0) / (sc[15] + 1.0)),
        "log_growth_R_10": math.log((rc[20] + 1.0) / (rc[10] + 1.0)),
        "log_growth_S_10": math.log((sc[20] + 1.0) / (sc[10] + 1.0)),
        "recent_inc_diff": rinc[3] - sinc[3],
        "log_recent_inc_ratio": math.log((max(0.0, rinc[3]) + 1.0) / (max(0.0, sinc[3]) + 1.0)),
        "active_case_frac_R": r["active"] / (rc[20] + 1.0),
        "active_case_frac_S": s["active"] / (sc[20] + 1.0),
        "accel_R": rinc[3] - rinc[2],
        "accel_S": sinc[3] - sinc[2],
        "intervention_present": vax["present"],
        "vax_efficacy": vax["efficacy"],
        "vax_coverage": vax["coverage"],
        "vax_day": vax["day"],
        "efficacy_x_coverage": effcov,
        "days_snapshot_to_vax": days_to_vax,
        "vax_already_started": already,
        "intervention_exposure_days": exposure,
        "effcov_x_exposure_days": effcov * exposure,
    }
    return row


def frame(corpus: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([feature_row(x) for x in corpus]).sort_values(["scenario_id", "kind", "horizon"]).reset_index(drop=True)


def base_rate_fit(train: pd.DataFrame) -> dict[tuple[str, int], float]:
    out = {}
    for (kind, horizon), g in train.groupby(["kind", "horizon"]):
        out[(str(kind), int(horizon))] = float((g.y.sum() + 1.0) / (len(g) + 2.0))
    return out


def base_rate_predict(df: pd.DataFrame, rates) -> np.ndarray:
    return clip_prob([rates[(str(k), int(h))] for k, h in zip(df.kind, df.horizon)])


def make_model(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, solver="lbfgs", max_iter=5000, random_state=BOOT_SEED)),
    ])


def fit_family(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> tuple[Pipeline, dict]:
    rows = []
    models = {}
    for C in CS:
        m = make_model(C)
        m.fit(train[features], train.y)
        pv = clip_prob(m.predict_proba(valid[features])[:, 1])
        rows.append({"C": C, "validation_brier": brier(valid.y, pv), "validation_log_loss": safe_log_loss(valid.y, pv)})
        models[C] = m
    best = min(rows, key=lambda x: (x["validation_brier"], x["C"]))
    return models[best["C"]], {"selected_C": best["C"], "candidates": rows}


def cluster_relative_lifts(df: pd.DataFrame, baseline_col: str, challenger_col: str) -> np.ndarray:
    vals = []
    for _, g in df.groupby("scenario_id"):
        bb = brier(g.y, g[baseline_col])
        cb = brier(g.y, g[challenger_col])
        if bb > 0:
            vals.append((bb - cb) / bb)
    return np.asarray(vals, dtype=float)


def bootstrap(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": None, "median": None, "lower": None, "upper": None, "n_clusters": 0}
    rng = np.random.default_rng(BOOT_SEED)
    sims = np.empty(BOOT, dtype=float)
    for i in range(BOOT):
        sims[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    return {
        "mean": float(vals.mean()), "median": float(np.median(vals)),
        "lower": float(np.quantile(sims, 0.025)), "upper": float(np.quantile(sims, 0.975)),
        "n_clusters": int(len(vals)),
    }


def metrics(df: pd.DataFrame, pcol: str) -> dict:
    y, p = df.y.to_numpy(), df[pcol].to_numpy()
    return {
        "n": int(len(df)),
        "scenarios": int(df.scenario_id.nunique()),
        "brier": brier(y, p),
        "log_loss": safe_log_loss(y, p),
        "ece10": ece10(y, p),
        "mean_probability": float(np.mean(p)),
        "base_rate": float(np.mean(y)),
        "calibration": calibration_intercept_slope(y, p),
    }


def relative_improvement(base: float, challenger: float) -> float:
    return float((base - challenger) / base)


def source_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main(fbsim_repo: Path, outdir: Path):
    scenarios = sample_scenarios(N_SCENARIOS, SCENARIO_SEED)
    if [int(x["scenario_id"]) for x in scenarios] != list(range(N_SCENARIOS)):
        raise RuntimeError("Unexpected scenario IDs")

    # Build TRAIN and VALIDATION first. HOLDOUT simulations are deliberately not built until
    # model family choices and router decisions have been frozen in memory from validation.
    train_corpus = build_corpus([s for s in scenarios if int(s["scenario_id"]) in TRAIN_IDS])
    valid_corpus = build_corpus([s for s in scenarios if int(s["scenario_id"]) in VALID_IDS])
    train = frame(train_corpus)
    valid = frame(valid_corpus)

    expected_train, expected_valid = 240, 120
    if len(train) != expected_train or len(valid) != expected_valid:
        raise RuntimeError(f"Unexpected train/validation question counts {len(train)}/{len(valid)}")

    b0_rates = base_rate_fit(train)
    m1, meta1 = fit_family(train, valid, B1_FEATURES)
    m2, meta2 = fit_family(train, valid, B1_FEATURES + B2_EXTRA)
    m3, meta3 = fit_family(train, valid, B1_FEATURES + B2_EXTRA + B3_EXTRA)

    valid = valid.copy()
    valid["p_B0"] = base_rate_predict(valid, b0_rates)
    valid["p_B1"] = clip_prob(m1.predict_proba(valid[B1_FEATURES])[:, 1])
    valid["p_B2"] = clip_prob(m2.predict_proba(valid[B1_FEATURES + B2_EXTRA])[:, 1])
    valid["p_B3"] = clip_prob(m3.predict_proba(valid[B1_FEATURES + B2_EXTRA + B3_EXTRA])[:, 1])

    router = {}
    for (kind, horizon), g in valid.groupby(["kind", "horizon"]):
        vals = cluster_relative_lifts(g, "p_B1", "p_B3")
        ci = bootstrap(vals)
        b1b, b3b = brier(g.y, g.p_B1), brier(g.y, g.p_B3)
        lift = relative_improvement(b1b, b3b)
        router[f"{kind}|{int(horizon)}"] = {
            "promoted": bool(lift > 0 and ci["lower"] is not None and ci["lower"] > 0),
            "validation_B1_brier": b1b,
            "validation_B3_brier": b3b,
            "validation_relative_brier_improvement": lift,
            "validation_scenario_bootstrap": ci,
        }

    # Only now materialize and resolve the untouched HOLDOUT worlds.
    holdout_corpus = build_corpus([s for s in scenarios if int(s["scenario_id"]) in HOLDOUT_IDS])
    hold = frame(holdout_corpus)
    if len(hold) != 120:
        raise RuntimeError(f"Unexpected holdout question count {len(hold)}")

    hold["p_B0"] = base_rate_predict(hold, b0_rates)
    hold["p_B1"] = clip_prob(m1.predict_proba(hold[B1_FEATURES])[:, 1])
    hold["p_B2"] = clip_prob(m2.predict_proba(hold[B1_FEATURES + B2_EXTRA])[:, 1])
    hold["p_B3"] = clip_prob(m3.predict_proba(hold[B1_FEATURES + B2_EXTRA + B3_EXTRA])[:, 1])
    hold["router_promoted"] = [router[f"{k}|{int(h)}"]["promoted"] for k, h in zip(hold.kind, hold.horizon)]
    hold["p_routed"] = np.where(hold.router_promoted, hold.p_B3, hold.p_B1)

    all_metrics = {name: metrics(hold, col) for name, col in [
        ("B0_reference_class", "p_B0"),
        ("B1_current_state", "p_B1"),
        ("B2_fast_trajectory", "p_B2"),
        ("B3_intervention_aware", "p_B3"),
        ("routed_policy", "p_routed"),
    ]}

    b1_b = all_metrics["B1_current_state"]["brier"]
    routed_b = all_metrics["routed_policy"]["brier"]
    aggregate_lift = relative_improvement(b1_b, routed_b)
    aggregate_ci = bootstrap(cluster_relative_lifts(hold, "p_B1", "p_routed"))

    cond = hold[hold.kind == "conditional"].copy()
    uncond = hold[hold.kind == "unconditional"].copy()
    cond_lift = relative_improvement(brier(cond.y, cond.p_B1), brier(cond.y, cond.p_routed))
    cond_ci = bootstrap(cluster_relative_lifts(cond, "p_B1", "p_routed"))
    uncond_lift = relative_improvement(brier(uncond.y, uncond.p_B1), brier(uncond.y, uncond.p_routed))

    conditions = {
        "aggregate_brier_improvement_ge_5pct": aggregate_lift >= 0.05,
        "aggregate_bootstrap_lower_gt_zero": aggregate_ci["lower"] is not None and aggregate_ci["lower"] > 0,
        "conditional_brier_improvement_ge_5pct": cond_lift >= 0.05,
        "conditional_bootstrap_lower_gt_zero": cond_ci["lower"] is not None and cond_ci["lower"] > 0,
        "unconditional_degradation_no_worse_than_2pct": uncond_lift >= -0.02,
        "routed_log_loss_no_worse_than_B1": all_metrics["routed_policy"]["log_loss"] <= all_metrics["B1_current_state"]["log_loss"],
        "forecast_coverage_100pct": bool(hold.p_routed.notna().all() and len(hold) == 120),
    }
    if all(conditions.values()):
        verdict = "CONTROLLED_WORLD_PASS"
    elif aggregate_lift > 0 and aggregate_ci["lower"] is not None and aggregate_ci["lower"] > 0:
        verdict = "CONTROLLED_WORLD_DIRECTIONAL"
    else:
        verdict = "CONTROLLED_WORLD_FAIL"

    by_cell = {}
    for (kind, horizon), g in hold.groupby(["kind", "horizon"]):
        key = f"{kind}|{int(horizon)}"
        by_cell[key] = {
            "router_promoted": bool(router[key]["promoted"]),
            "B1": metrics(g, "p_B1"),
            "B2": metrics(g, "p_B2"),
            "B3": metrics(g, "p_B3"),
            "routed": metrics(g, "p_routed"),
            "routed_relative_brier_improvement_vs_B1": relative_improvement(brier(g.y, g.p_B1), brier(g.y, g.p_routed)),
            "ungated_B3_relative_brier_improvement_vs_B1": relative_improvement(brier(g.y, g.p_B1), brier(g.y, g.p_B3)),
        }

    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_CONTROLLED_HOLDOUT_RESULT",
        "source": {
            "repo": "forecastingresearch/forecastbench-sim",
            "expected_commit": "a5b446cdbe6302302bed51616d0ced3b3f5239ed",
            "actual_commit": source_commit(fbsim_repo),
        },
        "scenario_seed": SCENARIO_SEED,
        "splits": {
            "train_scenarios": 60, "validation_scenarios": 30, "holdout_scenarios": 30,
            "train_questions": len(train), "validation_questions": len(valid), "holdout_questions": len(hold),
        },
        "model_selection": {"B1": meta1, "B2": meta2, "B3": meta3},
        "B0_rates": {f"{k[0]}|{k[1]}": v for k, v in b0_rates.items()},
        "router": router,
        "holdout_metrics": all_metrics,
        "holdout_by_cell": by_cell,
        "primary": {
            "aggregate_relative_brier_improvement_vs_B1": aggregate_lift,
            "aggregate_scenario_bootstrap": aggregate_ci,
            "conditional_relative_brier_improvement_vs_B1": cond_lift,
            "conditional_scenario_bootstrap": cond_ci,
            "unconditional_relative_brier_improvement_vs_B1": uncond_lift,
            "conditions": conditions,
            "verdict": verdict,
        },
        "ablation_descriptive": {
            "B2_vs_B1_relative_brier_improvement": relative_improvement(b1_b, all_metrics["B2_fast_trajectory"]["brier"]),
            "B3_vs_B2_conditional_relative_brier_improvement": relative_improvement(brier(cond.y, cond.p_B2), brier(cond.y, cond.p_B3)),
            "router_vs_ungated_B3_relative_brier_improvement": relative_improvement(all_metrics["B3_intervention_aware"]["brier"], routed_b),
        },
        "authority": "CONTROLLED_SIMULATOR_V4_DIAGNOSTIC_NOT_REAL_WORLD_F2",
    }

    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "FBSIM_PANDEMIC_V1_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    hold.to_csv(outdir / "FBSIM_PANDEMIC_V1_HOLDOUT.csv.gz", index=False, compression="gzip")
    valid.to_csv(outdir / "FBSIM_PANDEMIC_V1_VALIDATION.csv.gz", index=False, compression="gzip")

    lines = [
        "# PFSCE v8.2 ForecastBench-Sim Pandemic Validation",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        f"**Verdict: {verdict}.**",
        "",
        f"B1 holdout Brier: **{b1_b:.6f}**",
        f"Routed holdout Brier: **{routed_b:.6f}**",
        f"Aggregate relative Brier improvement: **{aggregate_lift*100:.3f}%**",
        f"Scenario-bootstrap 95% CI: **[{aggregate_ci['lower']*100:.3f}%, {aggregate_ci['upper']*100:.3f}%]**",
        f"Conditional relative improvement: **{cond_lift*100:.3f}%**",
        f"Conditional scenario-bootstrap 95% CI: **[{cond_ci['lower']*100:.3f}%, {cond_ci['upper']*100:.3f}%]**",
        f"Unconditional relative improvement: **{uncond_lift*100:.3f}%**",
        "",
        "## Frozen gate",
        *[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in conditions.items()],
        "",
        "This result validates only the controlled-world architectural slice described by the frozen protocol. It cannot establish real-world F2 authority.",
    ]
    (outdir / "FBSIM_PANDEMIC_V1_VERDICT.md").write_text("\n".join(lines) + "\n")
    hashes = {p.name: sha256(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    (outdir / "SHA256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))

    print("PFSCE_FBSIM_PANDEMIC_RESULT_BEGIN")
    print(json.dumps(result, indent=2))
    print("PFSCE_FBSIM_PANDEMIC_RESULT_END")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbsim-repo", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("experiment_output/fbsim_pandemic_v1"))
    args = ap.parse_args()
    main(args.fbsim_repo, args.out_dir)
