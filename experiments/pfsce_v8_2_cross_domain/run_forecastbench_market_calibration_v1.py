from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROTOCOL_ID = "PFSCE-V8.2-FORECASTBENCH-MARKET-CAL-V1"
EXPECTED_DATASET_COMMIT = "638d1e6808a0aa352851949bea62918fb55ca054"
MARKET_SOURCES = {"infer", "kalshi", "manifold", "metaculus", "polymarket"}
CS = [0.01, 0.1, 1.0, 10.0]
BOOT = 5000
SEED = 20260820
PCLIP = (0.005, 0.995)
TRAIN_END = pd.Timestamp("2025-06-30")
VALID_START = pd.Timestamp("2025-07-01")
VALID_END = pd.Timestamp("2025-12-31")
HOLD_START = pd.Timestamp("2026-01-01")
HOLD_END = pd.Timestamp("2026-06-30")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), PCLIP[0], PCLIP[1])


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1 - p))


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
    out = 0.0
    for i in range(10):
        mask = (p >= edges[i]) & (p <= edges[i + 1] if i == 9 else p < edges[i + 1])
        if np.any(mask):
            out += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def calibration(y, p):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return {"intercept": None, "slope": None}
    z = logit(p).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    m.fit(z, y)
    return {"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0, 0])}


def rel_improve(base: float, candidate: float) -> float:
    return float((base - candidate) / base) if base > 0 else float("nan")


def json_direction(v) -> str:
    if v is None:
        return "null"
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def due_from_resolution_name(path: Path) -> str | None:
    name = path.name
    if len(name) >= 10 and name[:10].count("-") == 2:
        return name[:10]
    return None


def horizon_bucket(days: float) -> str:
    if days <= 30:
        return "le30"
    if days <= 90:
        return "31_90"
    return "gt90"


def load_archive(repo: Path) -> tuple[pd.DataFrame, dict]:
    qdir = repo / "datasets" / "question_sets"
    rdir = repo / "datasets" / "resolution_sets"
    resolution_files = sorted(rdir.glob("*_resolution_set.json"))
    if not resolution_files:
        raise RuntimeError("No ForecastBench resolution sets found")

    provenance = []
    exclusions = Counter()
    rows = []
    paired = 0
    unpaired = 0

    for rp in resolution_files:
        due_hint = due_from_resolution_name(rp)
        try:
            rd = read_json(rp)
        except Exception as e:
            exclusions["resolution_file_unreadable"] += 1
            provenance.append({"file": str(rp.relative_to(repo)), "sha256": sha256(rp), "error": repr(e)})
            continue
        due = str(rd.get("forecast_due_date") or due_hint or "")
        qp = qdir / f"{due}-llm.json"
        prov = {"resolution_file": str(rp.relative_to(repo)), "resolution_sha256": sha256(rp), "forecast_due_date": due}
        if not qp.exists():
            unpaired += 1
            exclusions["missing_paired_question_set"] += 1
            provenance.append(prov)
            continue
        try:
            qd = read_json(qp)
        except Exception as e:
            unpaired += 1
            exclusions["question_file_unreadable"] += 1
            prov.update({"question_file": str(qp.relative_to(repo)), "question_sha256": sha256(qp), "error": repr(e)})
            provenance.append(prov)
            continue
        prov.update({"question_file": str(qp.relative_to(repo)), "question_sha256": sha256(qp)})
        provenance.append(prov)
        paired += 1

        questions = qd.get("questions") or []
        resolutions = rd.get("resolutions") or []
        if not isinstance(questions, list) or not isinstance(resolutions, list):
            exclusions["bad_top_level_schema"] += 1
            continue

        qmap = {}
        for q in questions:
            if not isinstance(q, dict):
                exclusions["question_not_object"] += 1
                continue
            qid = q.get("id")
            src = str(q.get("source") or "")
            if src not in MARKET_SOURCES:
                exclusions["non_market_source"] += 1
                continue
            if not isinstance(qid, str):
                exclusions["non_string_question_id"] += 1
                continue
            fv = q.get("freeze_datetime_value")
            try:
                fv = float(fv)
            except (TypeError, ValueError):
                exclusions["non_numeric_freeze_value"] += 1
                continue
            if not (0.0 < fv < 1.0):
                exclusions["freeze_value_not_probability"] += 1
                continue
            qmap[(qid, src)] = fv

        for rr in resolutions:
            if not isinstance(rr, dict):
                exclusions["resolution_not_object"] += 1
                continue
            qid = rr.get("id")
            src = str(rr.get("source") or "")
            if src not in MARKET_SOURCES:
                exclusions["resolution_non_market_source"] += 1
                continue
            if not isinstance(qid, str):
                exclusions["resolution_non_string_id"] += 1
                continue
            key = (qid, src)
            if key not in qmap:
                exclusions["no_eligible_question_match"] += 1
                continue
            if "resolved" in rr and not bool(rr.get("resolved")):
                exclusions["unresolved"] += 1
                continue
            try:
                y = float(rr.get("resolved_to"))
            except (TypeError, ValueError):
                exclusions["non_numeric_resolution"] += 1
                continue
            if y not in (0.0, 1.0):
                exclusions["non_binary_resolution"] += 1
                continue
            forecast_due = pd.to_datetime(due, errors="coerce")
            res_date = pd.to_datetime(rr.get("resolution_date"), errors="coerce")
            if pd.isna(forecast_due) or pd.isna(res_date):
                exclusions["unparseable_date"] += 1
                continue
            if res_date < forecast_due:
                exclusions["resolution_before_due"] += 1
                continue
            days = float((res_date - forecast_due).days)
            rows.append({
                "id": qid,
                "source": src,
                "forecast_due_date": forecast_due,
                "resolution_date": res_date,
                "direction": json_direction(rr.get("direction")),
                "y": int(y),
                "p_market": float(qmap[key]),
                "days_to_resolution": days,
                "horizon_bucket": horizon_bucket(days),
                "origin_resolution_file": str(rp.relative_to(repo)),
                "origin_question_file": str(qp.relative_to(repo)),
            })

    if not rows:
        raise RuntimeError("No eligible ForecastBench market targets found")
    df = pd.DataFrame(rows)
    pre = len(df)
    df = df.sort_values(["forecast_due_date", "origin_resolution_file", "id", "resolution_date", "source", "direction"])
    dedup_cols = ["id", "forecast_due_date", "resolution_date", "source", "direction"]
    df = df.drop_duplicates(dedup_cols, keep="first").reset_index(drop=True)
    exclusions["exact_or_repeated_target_duplicates"] += pre - len(df)

    meta = {
        "paired_resolution_question_sets": paired,
        "unpaired_resolution_sets": unpaired,
        "resolution_files_seen": len(resolution_files),
        "eligible_rows_after_dedup": int(len(df)),
        "exclusions": dict(exclusions),
        "input_files": provenance,
    }
    return df, meta


def cohort(df: pd.DataFrame, which: str) -> pd.DataFrame:
    d = df.forecast_due_date
    if which == "train":
        return df[d <= TRAIN_END].copy()
    if which == "validation":
        return df[(d >= VALID_START) & (d <= VALID_END)].copy()
    if which == "holdout":
        return df[(d >= HOLD_START) & (d <= HOLD_END)].copy()
    raise ValueError(which)


def prepare_features(train: pd.DataFrame, other: pd.DataFrame | None = None):
    def build(d: pd.DataFrame) -> pd.DataFrame:
        z = pd.DataFrame(index=d.index)
        z["market_logit"] = logit(d.p_market)
        z["abs_market_logit"] = np.abs(z.market_logit)
        z["log_days"] = np.log1p(np.clip(d.days_to_resolution.to_numpy(float), 0, None))
        src = pd.get_dummies(d.source, prefix="src", dtype=float)
        hor = pd.get_dummies(d.horizon_bucket, prefix="hor", dtype=float)
        z = pd.concat([z, src, hor], axis=1)
        for c in src.columns:
            z[f"{c}_x_logit"] = src[c] * z.market_logit
        for c in hor.columns:
            z[f"{c}_x_logit"] = hor[c] * z.market_logit
        return z.astype(float)

    Xtr = build(train)
    if other is None:
        return Xtr
    Xo = build(other).reindex(columns=Xtr.columns, fill_value=0.0)
    return Xtr, Xo


def model(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, solver="lbfgs", max_iter=5000, random_state=SEED)),
    ])


def select_model(train: pd.DataFrame, valid: pd.DataFrame, feature_kind: str):
    if feature_kind == "M1":
        Xtr = logit(train.p_market).reshape(-1, 1)
        Xv = logit(valid.p_market).reshape(-1, 1)
    elif feature_kind == "M2":
        Xtr, Xv = prepare_features(train, valid)
    else:
        raise ValueError(feature_kind)
    candidates = []
    models = {}
    for C in CS:
        m = model(C)
        m.fit(Xtr, train.y)
        pv = clip_prob(m.predict_proba(Xv)[:, 1])
        candidates.append({"C": C, "validation_brier": brier(valid.y, pv), "validation_log_loss": safe_log_loss(valid.y, pv)})
        models[C] = m
    best = min(candidates, key=lambda r: (r["validation_brier"], r["C"]))
    return models[best["C"]], {"selected_C": best["C"], "candidates": candidates}


def predict_selected(m, train: pd.DataFrame, target: pd.DataFrame, feature_kind: str):
    if feature_kind == "M1":
        X = logit(target.p_market).reshape(-1, 1)
    else:
        _, X = prepare_features(train, target)
    return clip_prob(m.predict_proba(X)[:, 1])


def cluster_bootstrap(df: pd.DataFrame, baseline_col: str, challenger_col: str) -> dict:
    clusters = []
    for qid, g in df.groupby("id", sort=True):
        y = g.y.to_numpy(float)
        b = clip_prob(g[baseline_col])
        c = clip_prob(g[challenger_col])
        clusters.append((str(qid), float(np.sum((b-y)**2)), float(np.sum((c-y)**2)), int(len(g))))
    if not clusters:
        return {"observed": None, "mean": None, "median": None, "lower": None, "upper": None, "n_clusters": 0}
    bs = np.array([x[1] for x in clusters], float)
    cs = np.array([x[2] for x in clusters], float)
    ns = np.array([x[3] for x in clusters], float)
    observed_b = bs.sum()/ns.sum(); observed_c = cs.sum()/ns.sum()
    observed = rel_improve(observed_b, observed_c)
    rng = np.random.default_rng(SEED)
    n = len(clusters); sims = np.empty(BOOT, float)
    for i in range(BOOT):
        ix = rng.integers(0, n, size=n)
        bb = bs[ix].sum()/ns[ix].sum(); cc = cs[ix].sum()/ns[ix].sum()
        sims[i] = rel_improve(bb, cc)
    return {"observed": observed, "mean": float(sims.mean()), "median": float(np.median(sims)), "lower": float(np.quantile(sims, .025)), "upper": float(np.quantile(sims, .975)), "n_clusters": int(n), "bootstrap_draws": BOOT}


def metrics(df: pd.DataFrame, pcol: str) -> dict:
    return {
        "n_rows": int(len(df)),
        "unique_questions": int(df.id.nunique()),
        "brier": brier(df.y, df[pcol]),
        "log_loss": safe_log_loss(df.y, df[pcol]),
        "ece10": ece10(df.y, df[pcol]),
        "mean_probability": float(np.mean(df[pcol])),
        "base_rate": float(np.mean(df.y)),
        "calibration": calibration(df.y, df[pcol]),
    }


def summarize_cohort(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "unique_questions": int(df.id.nunique()),
        "due_min": str(df.forecast_due_date.min().date()) if len(df) else None,
        "due_max": str(df.forecast_due_date.max().date()) if len(df) else None,
        "positive_rows": int(df.y.sum()) if len(df) else 0,
        "positive_rate": float(df.y.mean()) if len(df) else None,
        "sources": {str(k): int(v) for k, v in df.source.value_counts().sort_index().items()},
    }


def main(repo: Path, outdir: Path):
    actual_commit = git_head(repo)
    if actual_commit != EXPECTED_DATASET_COMMIT:
        raise RuntimeError(f"Dataset commit mismatch expected={EXPECTED_DATASET_COMMIT} actual={actual_commit}")

    all_df, archive_meta = load_archive(repo)
    train = cohort(all_df, "train")
    valid = cohort(all_df, "validation")
    # Do not access HOLDOUT rows for fitting/router selection. Materialize only after choices are frozen.
    if min(len(train), len(valid)) == 0 or min(train.y.nunique(), valid.y.nunique()) < 2:
        raise RuntimeError(f"Insufficient train/validation data train={len(train)} valid={len(valid)}")

    m1, m1meta = select_model(train, valid, "M1")
    m2, m2meta = select_model(train, valid, "M2")
    valid = valid.copy()
    valid["p_M0"] = clip_prob(valid.p_market)
    valid["p_M1"] = predict_selected(m1, train, valid, "M1")
    valid["p_M2"] = predict_selected(m2, train, valid, "M2")

    router = {}
    for (src, hb), g in valid.groupby(["source", "horizon_bucket"]):
        uq = int(g.id.nunique()); pos = int(g.y.sum()); neg = int(len(g)-g.y.sum())
        lift = rel_improve(brier(g.y, g.p_M0), brier(g.y, g.p_M2))
        ci = cluster_bootstrap(g, "p_M0", "p_M2")
        conditions = {
            "unique_questions_ge_50": uq >= 50,
            "positives_ge_5": pos >= 5,
            "negatives_ge_5": neg >= 5,
            "observed_lift_gt_zero": bool(np.isfinite(lift) and lift > 0),
            "bootstrap_lower_gt_zero": ci["lower"] is not None and ci["lower"] > 0,
        }
        router[f"{src}|{hb}"] = {
            "promoted": all(conditions.values()), "conditions": conditions,
            "n_rows": int(len(g)), "unique_questions": uq, "positive_rows": pos, "negative_rows": neg,
            "validation_M0_brier": brier(g.y, g.p_M0), "validation_M2_brier": brier(g.y, g.p_M2),
            "validation_relative_brier_improvement": lift, "validation_question_bootstrap": ci,
        }

    # Choices are now frozen. Slice the untouched historical holdout.
    hold = cohort(all_df, "holdout")
    if len(hold) == 0 or hold.y.nunique() < 2:
        raise RuntimeError(f"Insufficient holdout data n={len(hold)} classes={hold.y.nunique()}")
    hold = hold.copy()
    hold["p_M0"] = clip_prob(hold.p_market)
    hold["p_M1"] = predict_selected(m1, train, hold, "M1")
    hold["p_M2"] = predict_selected(m2, train, hold, "M2")
    hold["router_cell"] = hold.source.astype(str) + "|" + hold.horizon_bucket.astype(str)
    hold["router_promoted"] = hold.router_cell.map(lambda c: bool(router.get(c, {}).get("promoted", False)))
    hold["p_routed"] = np.where(hold.router_promoted, hold.p_M2, hold.p_M0)

    mm = {name: metrics(hold, col) for name, col in [("M0_frozen_market", "p_M0"), ("M1_platt", "p_M1"), ("M2_metadata", "p_M2"), ("routed_policy", "p_routed")]}
    agg_lift = rel_improve(mm["M0_frozen_market"]["brier"], mm["routed_policy"]["brier"])
    agg_ci = cluster_bootstrap(hold, "p_M0", "p_routed")

    by_source = {}
    source_safety = True
    for src, g in hold.groupby("source"):
        unique_n = int(g.id.nunique())
        lift = rel_improve(brier(g.y, g.p_M0), brier(g.y, g.p_routed))
        bad = bool(unique_n >= 100 and lift < -0.05)
        source_safety = source_safety and not bad
        by_source[str(src)] = {"unique_questions": unique_n, "n_rows": int(len(g)), "relative_brier_improvement": lift, "safety_violation": bad, "M0": metrics(g, "p_M0"), "routed": metrics(g, "p_routed")}

    by_cell = {}
    for (src, hb), g in hold.groupby(["source", "horizon_bucket"]):
        cell = f"{src}|{hb}"
        by_cell[cell] = {
            "router_promoted_from_validation": bool(router.get(cell, {}).get("promoted", False)),
            "n_rows": int(len(g)), "unique_questions": int(g.id.nunique()),
            "M0_brier": brier(g.y, g.p_M0), "M2_brier": brier(g.y, g.p_M2), "routed_brier": brier(g.y, g.p_routed),
            "M2_relative_brier_improvement": rel_improve(brier(g.y, g.p_M0), brier(g.y, g.p_M2)),
            "routed_relative_brier_improvement": rel_improve(brier(g.y, g.p_M0), brier(g.y, g.p_routed)),
        }

    coverage = float(hold.p_routed.notna().mean())
    conditions = {
        "aggregate_brier_improvement_ge_2pct": agg_lift >= 0.02,
        "bootstrap_lower_gt_zero": agg_ci["lower"] is not None and agg_ci["lower"] > 0,
        "routed_log_loss_no_worse_than_market": mm["routed_policy"]["log_loss"] <= mm["M0_frozen_market"]["log_loss"],
        "forecast_coverage_100pct": coverage == 1.0,
        "no_large_source_degradation": bool(source_safety),
    }
    if all(conditions.values()): verdict = "BROAD_CALIBRATION_PASS"
    elif agg_lift > 0 and agg_ci["lower"] is not None and agg_ci["lower"] > 0: verdict = "BROAD_CALIBRATION_DIRECTIONAL"
    else: verdict = "BROAD_CALIBRATION_FAIL"

    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_RETROSPECTIVE_HOLDOUT_RESULT",
        "dataset": {"repo": "forecastingresearch/forecastbench-datasets", "commit": actual_commit},
        "baseline_semantics": "published freeze_datetime_value, approximately 10 days before forecast_due_date; not due-date market price",
        "archive": archive_meta,
        "cohorts": {"train": summarize_cohort(train), "validation": summarize_cohort(valid), "holdout": summarize_cohort(hold)},
        "model_selection": {"M1": m1meta, "M2": m2meta},
        "router": router,
        "promoted_cells": [k for k,v in router.items() if v["promoted"]],
        "holdout_metrics": mm,
        "holdout_by_source": by_source,
        "holdout_by_cell": by_cell,
        "primary": {
            "aggregate_relative_brier_improvement": agg_lift,
            "aggregate_question_cluster_bootstrap": agg_ci,
            "coverage": coverage,
            "promoted_row_share": float(hold.router_promoted.mean()),
            "fallback_row_share": float((~hold.router_promoted).mean()),
            "conditions": conditions,
            "verdict": verdict,
        },
        "authority": "RETROSPECTIVE_BROAD_CALIBRATION_V1_NOT_PROSPECTIVE_F2",
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "FORECASTBENCH_MARKET_CAL_V1_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    hold.to_csv(outdir / "FORECASTBENCH_MARKET_CAL_V1_HOLDOUT.csv.gz", index=False, compression="gzip")
    valid.to_csv(outdir / "FORECASTBENCH_MARKET_CAL_V1_VALIDATION.csv.gz", index=False, compression="gzip")
    (outdir / "FORECASTBENCH_MARKET_CAL_V1_INPUT_PROVENANCE.json").write_text(json.dumps(archive_meta, indent=2, sort_keys=True))
    lines = [
        "# PFSCE v8.2 ForecastBench Broad Market Calibration", "",
        f"Protocol: `{PROTOCOL_ID}`", f"**Verdict: {verdict}.**", "",
        f"Holdout rows / unique questions: **{len(hold)} / {hold.id.nunique()}**",
        f"M0 frozen-market Brier: **{mm['M0_frozen_market']['brier']:.6f}**",
        f"Routed Brier: **{mm['routed_policy']['brier']:.6f}**",
        f"Relative Brier improvement: **{agg_lift*100:.3f}%**",
        f"Question-cluster 95% CI: **[{agg_ci['lower']*100:.3f}%, {agg_ci['upper']*100:.3f}%]**",
        f"Promoted row share: **{hold.router_promoted.mean()*100:.2f}%**", "",
        "## Frozen gate", *[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k,v in conditions.items()], "",
        "The baseline is the public market probability frozen about ten days before the ForecastBench forecast deadline; this is not evidence of beating the contemporaneous due-date market.",
    ]
    (outdir / "FORECASTBENCH_MARKET_CAL_V1_VERDICT.md").write_text("\n".join(lines)+"\n")
    hashes = {p.name: sha256(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    (outdir / "SHA256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))

    print("PFSCE_FORECASTBENCH_MARKET_CAL_RESULT_BEGIN")
    print(json.dumps(result, indent=2))
    print("PFSCE_FORECASTBENCH_MARKET_CAL_RESULT_END")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-repo", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("experiment_output/forecastbench_market_cal_v1"))
    args = ap.parse_args()
    main(args.dataset_repo, args.out_dir)
