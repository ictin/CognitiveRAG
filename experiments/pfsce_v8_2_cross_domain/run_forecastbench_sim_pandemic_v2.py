from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_forecastbench_sim_pandemic_v1 as base

PROTOCOL_ID = "PFSCE-V8.2-FBSIM-PANDEMIC-V2"
SEED = 20260821
BOOT = 5000
N_SCENARIOS = 600
TRAIN_IDS = set(range(0, 300))
VALID_IDS = set(range(300, 450))
HOLDOUT_IDS = set(range(450, 600))
EXPECTED_COMMIT = "a5b446cdbe6302302bed51616d0ced3b3f5239ed"

# Make inherited deterministic model construction use the frozen V2 seed.
base.BOOT_SEED = SEED


def aggregate_cluster_bootstrap(df: pd.DataFrame, baseline_col: str, challenger_col: str) -> dict:
    """Bootstrap scenario clusters and recompute aggregate Brier ratio per draw.

    This is intentionally different from V1's mean of within-scenario ratios.
    """
    cluster_rows = []
    for scenario_id, g in df.groupby("scenario_id", sort=True):
        y = g.y.to_numpy(float)
        b = base.clip_prob(g[baseline_col].to_numpy(float))
        c = base.clip_prob(g[challenger_col].to_numpy(float))
        cluster_rows.append((int(scenario_id), float(np.sum((b - y) ** 2)), float(np.sum((c - y) ** 2)), int(len(g))))
    if not cluster_rows:
        return {"observed": None, "mean": None, "median": None, "lower": None, "upper": None, "n_clusters": 0}

    ids = np.array([r[0] for r in cluster_rows], dtype=int)
    bss = np.array([r[1] for r in cluster_rows], dtype=float)
    css = np.array([r[2] for r in cluster_rows], dtype=float)
    ns = np.array([r[3] for r in cluster_rows], dtype=float)
    total_b = bss.sum() / ns.sum()
    total_c = css.sum() / ns.sum()
    observed = float((total_b - total_c) / total_b) if total_b > 0 else None

    rng = np.random.default_rng(SEED)
    ncl = len(ids)
    sims = np.empty(BOOT, dtype=float)
    for i in range(BOOT):
        idx = rng.integers(0, ncl, size=ncl)
        bb = bss[idx].sum() / ns[idx].sum()
        cc = css[idx].sum() / ns[idx].sum()
        sims[i] = (bb - cc) / bb if bb > 0 else np.nan
    sims = sims[np.isfinite(sims)]
    return {
        "observed": observed,
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "lower": float(np.quantile(sims, 0.025)),
        "upper": float(np.quantile(sims, 0.975)),
        "n_clusters": int(ncl),
        "bootstrap_draws": int(len(sims)),
    }


def positive_counts(df: pd.DataFrame) -> dict:
    out = {"all": {"n": int(len(df)), "positive": int(df.y.sum()), "rate": float(df.y.mean()), "scenarios": int(df.scenario_id.nunique())}}
    for (kind, horizon), g in df.groupby(["kind", "horizon"]):
        out[f"{kind}|{int(horizon)}"] = {
            "n": int(len(g)), "positive": int(g.y.sum()), "rate": float(g.y.mean()), "scenarios": int(g.scenario_id.nunique())
        }
    return out


def main(fbsim_repo: Path, outdir: Path):
    scenarios = base.sample_scenarios(N_SCENARIOS, SEED)
    ids = [int(x["scenario_id"]) for x in scenarios]
    if ids != list(range(N_SCENARIOS)):
        raise RuntimeError("Unexpected V2 scenario IDs")

    # Resolve only TRAIN and VALIDATION first.
    train_corpus = base.build_corpus([s for s in scenarios if int(s["scenario_id"]) in TRAIN_IDS])
    valid_corpus = base.build_corpus([s for s in scenarios if int(s["scenario_id"]) in VALID_IDS])
    train = base.frame(train_corpus)
    valid = base.frame(valid_corpus)
    if len(train) != 1200 or len(valid) != 600:
        raise RuntimeError(f"Unexpected V2 train/validation counts {len(train)}/{len(valid)}")

    rates = base.base_rate_fit(train)
    m1, meta1 = base.fit_family(train, valid, base.B1_FEATURES)
    m2, meta2 = base.fit_family(train, valid, base.B1_FEATURES + base.B2_EXTRA)
    m3, meta3 = base.fit_family(train, valid, base.B1_FEATURES + base.B2_EXTRA + base.B3_EXTRA)

    valid = valid.copy()
    valid["p_B0"] = base.base_rate_predict(valid, rates)
    valid["p_B1"] = base.clip_prob(m1.predict_proba(valid[base.B1_FEATURES])[:, 1])
    valid["p_B2"] = base.clip_prob(m2.predict_proba(valid[base.B1_FEATURES + base.B2_EXTRA])[:, 1])
    valid["p_B3"] = base.clip_prob(m3.predict_proba(valid[base.B1_FEATURES + base.B2_EXTRA + base.B3_EXTRA])[:, 1])

    router = {}
    for (kind, horizon), g in valid.groupby(["kind", "horizon"]):
        b1b = base.brier(g.y, g.p_B1)
        b3b = base.brier(g.y, g.p_B3)
        lift = base.relative_improvement(b1b, b3b)
        ci = aggregate_cluster_bootstrap(g, "p_B1", "p_B3")
        router[f"{kind}|{int(horizon)}"] = {
            "promoted": bool(lift > 0 and ci["lower"] is not None and ci["lower"] > 0),
            "validation_B1_brier": b1b,
            "validation_B3_brier": b3b,
            "validation_relative_brier_improvement": lift,
            "validation_aggregate_scenario_bootstrap": ci,
        }

    # Freeze all choices in memory before any HOLDOUT world is simulated/resolved.
    hold_corpus = base.build_corpus([s for s in scenarios if int(s["scenario_id"]) in HOLDOUT_IDS])
    hold = base.frame(hold_corpus)
    if len(hold) != 600:
        raise RuntimeError(f"Unexpected V2 holdout count {len(hold)}")

    hold["p_B0"] = base.base_rate_predict(hold, rates)
    hold["p_B1"] = base.clip_prob(m1.predict_proba(hold[base.B1_FEATURES])[:, 1])
    hold["p_B2"] = base.clip_prob(m2.predict_proba(hold[base.B1_FEATURES + base.B2_EXTRA])[:, 1])
    hold["p_B3"] = base.clip_prob(m3.predict_proba(hold[base.B1_FEATURES + base.B2_EXTRA + base.B3_EXTRA])[:, 1])
    hold["router_promoted"] = [router[f"{k}|{int(h)}"]["promoted"] for k, h in zip(hold.kind, hold.horizon)]
    hold["p_routed"] = np.where(hold.router_promoted, hold.p_B3, hold.p_B1)

    all_metrics = {name: base.metrics(hold, col) for name, col in [
        ("B0_reference_class", "p_B0"),
        ("B1_current_state", "p_B1"),
        ("B2_fast_trajectory", "p_B2"),
        ("B3_intervention_aware", "p_B3"),
        ("routed_policy", "p_routed"),
    ]}

    b1_b = all_metrics["B1_current_state"]["brier"]
    routed_b = all_metrics["routed_policy"]["brier"]
    aggregate_lift = base.relative_improvement(b1_b, routed_b)
    aggregate_ci = aggregate_cluster_bootstrap(hold, "p_B1", "p_routed")

    cond = hold[hold.kind == "conditional"].copy()
    uncond = hold[hold.kind == "unconditional"].copy()
    cond_lift = base.relative_improvement(base.brier(cond.y, cond.p_B1), base.brier(cond.y, cond.p_routed))
    cond_ci = aggregate_cluster_bootstrap(cond, "p_B1", "p_routed")
    uncond_lift = base.relative_improvement(base.brier(uncond.y, uncond.p_B1), base.brier(uncond.y, uncond.p_routed))

    conditions = {
        "aggregate_brier_improvement_ge_5pct": aggregate_lift >= 0.05,
        "aggregate_bootstrap_lower_gt_zero": aggregate_ci["lower"] is not None and aggregate_ci["lower"] > 0,
        "conditional_brier_improvement_ge_5pct": cond_lift >= 0.05,
        "conditional_bootstrap_lower_gt_zero": cond_ci["lower"] is not None and cond_ci["lower"] > 0,
        "unconditional_degradation_no_worse_than_2pct": uncond_lift >= -0.02,
        "routed_log_loss_no_worse_than_B1": all_metrics["routed_policy"]["log_loss"] <= all_metrics["B1_current_state"]["log_loss"],
        "forecast_coverage_100pct": bool(hold.p_routed.notna().all() and len(hold) == 600),
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
            "B1": base.metrics(g, "p_B1"),
            "B2": base.metrics(g, "p_B2"),
            "B3": base.metrics(g, "p_B3"),
            "routed": base.metrics(g, "p_routed"),
            "routed_relative_brier_improvement_vs_B1": base.relative_improvement(base.brier(g.y, g.p_B1), base.brier(g.y, g.p_routed)),
            "routed_aggregate_scenario_bootstrap": aggregate_cluster_bootstrap(g, "p_B1", "p_routed"),
            "ungated_B3_relative_brier_improvement_vs_B1": base.relative_improvement(base.brier(g.y, g.p_B1), base.brier(g.y, g.p_B3)),
        }

    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_FRESH_COHORT_HOLDOUT_RESULT",
        "v1_status_preserved": "CONTROLLED_WORLD_FAIL",
        "source": {
            "repo": "forecastingresearch/forecastbench-sim",
            "expected_commit": EXPECTED_COMMIT,
            "actual_commit": base.source_commit(fbsim_repo),
        },
        "scenario_seed": SEED,
        "splits": {
            "train_scenarios": 300, "validation_scenarios": 150, "holdout_scenarios": 150,
            "train_questions": len(train), "validation_questions": len(valid), "holdout_questions": len(hold),
            "effective_independent_N": {"train": 300, "validation": 150, "holdout": 150},
        },
        "positive_class_counts": {
            "train": positive_counts(train),
            "validation": positive_counts(valid),
            "holdout": positive_counts(hold),
        },
        "model_selection": {"B1": meta1, "B2": meta2, "B3": meta3},
        "B0_rates": {f"{k[0]}|{k[1]}": v for k, v in rates.items()},
        "router": router,
        "promoted_cells": [k for k, v in router.items() if v["promoted"]],
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
            "B2_vs_B1_relative_brier_improvement": base.relative_improvement(b1_b, all_metrics["B2_fast_trajectory"]["brier"]),
            "B3_vs_B1_ungated_relative_brier_improvement": base.relative_improvement(b1_b, all_metrics["B3_intervention_aware"]["brier"]),
            "B3_vs_B2_conditional_relative_brier_improvement": base.relative_improvement(base.brier(cond.y, cond.p_B2), base.brier(cond.y, cond.p_B3)),
            "router_vs_ungated_B3_relative_brier_improvement": base.relative_improvement(all_metrics["B3_intervention_aware"]["brier"], routed_b),
        },
        "authority": "CONTROLLED_SIMULATOR_V4_REPLICATION_NOT_REAL_WORLD_F2",
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "FBSIM_PANDEMIC_V2_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    hold.to_csv(outdir / "FBSIM_PANDEMIC_V2_HOLDOUT.csv.gz", index=False, compression="gzip")
    valid.to_csv(outdir / "FBSIM_PANDEMIC_V2_VALIDATION.csv.gz", index=False, compression="gzip")

    lines = [
        "# PFSCE v8.2 ForecastBench-Sim Pandemic v2",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        f"**Verdict: {verdict}.**",
        "",
        f"B1 holdout Brier: **{b1_b:.6f}**",
        f"Routed holdout Brier: **{routed_b:.6f}**",
        f"Aggregate relative Brier improvement: **{aggregate_lift*100:.3f}%**",
        f"Aggregate scenario-bootstrap 95% CI: **[{aggregate_ci['lower']*100:.3f}%, {aggregate_ci['upper']*100:.3f}%]**",
        f"Conditional relative improvement: **{cond_lift*100:.3f}%**",
        f"Conditional 95% CI: **[{cond_ci['lower']*100:.3f}%, {cond_ci['upper']*100:.3f}%]**",
        f"Unconditional relative improvement: **{uncond_lift*100:.3f}%**",
        f"Promoted router cells: **{', '.join(result['promoted_cells']) if result['promoted_cells'] else 'none'}**",
        "",
        "## Frozen gate",
        *[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in conditions.items()],
        "",
        "V1 remains a frozen failure. V2 uses a fresh cohort and a preregistered correction to the cluster-bootstrap estimand. Neither result grants real-world F2 authority.",
    ]
    (outdir / "FBSIM_PANDEMIC_V2_VERDICT.md").write_text("\n".join(lines) + "\n")
    hashes = {p.name: base.sha256(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    (outdir / "SHA256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))

    print("PFSCE_FBSIM_PANDEMIC_V2_RESULT_BEGIN")
    print(json.dumps(result, indent=2))
    print("PFSCE_FBSIM_PANDEMIC_V2_RESULT_END")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbsim-repo", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("experiment_output/fbsim_pandemic_v2"))
    args = ap.parse_args()
    main(args.fbsim_repo, args.out_dir)
