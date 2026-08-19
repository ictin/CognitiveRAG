from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import run_b1 as b1

REPLICATION_ID = "NECF-001-WP2-REPLICATION-2025-v1"
PROTOCOL_FAMILY = "NECF-001-WP2-CONFIRMATORY-2024-v1"
SOURCE_RECORD_2025 = "18448416"
SOURCE_CREATED_2025 = "2026-02-01"
SOURCE_AUTHORITY = "V1_PROVISIONAL_REVISION_SENSITIVE_REPLICATION"
ARCHIVES_2025 = {
    "eia930-2025half1.zip": "4ebc17b3e786b5c4a06315afc285093f",
    "eia930-2025half2.zip": "9a7fe2bb3cdfe42236782ab7291a49ba",
}
EXPECTED_PROMOTED = {"CISO": True, "ERCO": False, "PJM": True, "ISNE": False}
EXPECTED_ALPHA = 100.0
SEED = 20260819
BOOT = 5000


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(7):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(30, 180),
                headers={"User-Agent": "PFSCE-NECF-replication/1.0"},
            ) as r:
                if r.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for c in r.iter_content(1024 * 1024):
                        if c:
                            f.write(c)
                tmp.replace(dest)
                return
        except Exception as e:
            if i == 6:
                raise
            print("RETRY", dest.name, i + 1, repr(e), flush=True)
            time.sleep(min(60, 2 ** (i + 1)))


def load_2025_target() -> tuple[pd.DataFrame, dict]:
    cache = Path("experiment_output/replication_2025/raw")
    frames = []
    provenance = []
    for fn, md5 in ARCHIVES_2025.items():
        p = cache / fn
        url = f"https://zenodo.org/records/{SOURCE_RECORD_2025}/files/{fn}?download=1"
        print("DOWNLOAD_2025", fn, flush=True)
        download(url, p)
        got = digest(p, "md5")
        if got != md5:
            raise RuntimeError(f"MD5 mismatch {fn} expected={md5} got={got}")
        d, members = b1.read_zip(p)
        d = d[(d.datetime_utc >= pd.Timestamp("2025-01-01T00:00:00Z")) &
              (d.datetime_utc <= pd.Timestamp("2025-12-31T23:59:59Z"))].copy()
        frames.append(d)
        provenance.append({
            "file": fn,
            "url": url,
            "expected_md5": md5,
            "md5": got,
            "sha256": digest(p, "sha256"),
            "size": p.stat().st_size,
            "selected_2025_rows": int(len(d)),
            "members": members,
        })
    x = pd.concat(frames, ignore_index=True)
    x = x.sort_values(["ba", "datetime_utc"]).drop_duplicates(["ba", "datetime_utc"], keep="last")
    return x, {
        "record": SOURCE_RECORD_2025,
        "published": SOURCE_CREATED_2025,
        "authority": SOURCE_AUTHORITY,
        "files": provenance,
        "rows_2025": int(len(x)),
    }


def weekly_cluster_bootstrap(df: pd.DataFrame, pred_col: str) -> dict:
    q = df.copy()
    q["week"] = q.datetime_utc.dt.tz_localize(None).dt.to_period("W").astype(str)
    vals = []
    rows = []
    for (ba, week), g in q.groupby(["ba_code", "week"]):
        baseline = float(np.mean(np.abs(g.demand - g.forecast)))
        challenger = float(np.mean(np.abs(g.demand - g[pred_col])))
        if baseline > 0:
            lift = (baseline - challenger) / baseline
            vals.append(lift)
            rows.append({"ba": str(ba), "week": week, "relative_lift": float(lift)})
    v = np.asarray(vals, float)
    rng = np.random.default_rng(SEED)
    sims = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(BOOT)])
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "lower": float(np.quantile(sims, 0.025)),
        "upper": float(np.quantile(sims, 0.975)),
        "n_clusters": int(len(v)),
        "clusters": rows,
    }


def main() -> None:
    out = Path("experiment_output/replication_2025")
    out.mkdir(parents=True, exist_ok=True)

    # Frozen 2020-2024 source. This is intentionally the same historical vintage used
    # by the 2024 confirmatory experiment; it must not be replaced by a later 2026 vintage.
    historical_2020_2024 = b1.load_data()
    target_2025, target_provenance = load_2025_target()

    historical_2020_2024 = historical_2020_2024[
        historical_2020_2024.datetime_utc <= pd.Timestamp("2024-12-31T23:59:59Z")
    ].copy()
    combined = pd.concat([historical_2020_2024, target_2025], ignore_index=True)
    combined = combined.sort_values(["ba", "datetime_utc"]).drop_duplicates(["ba", "datetime_utc"], keep="last")

    good, bad = b1.quarantine(combined)
    f = b1.target_filter(b1.features(good))
    F = b1.cols(f)

    train = b1.split(f, "2020-01-01T00:00:00Z", "2022-12-31T23:00:00Z").dropna(subset=["demand", "forecast", "residual"])
    val = b1.split(f, "2023-01-01T00:00:00Z", "2023-12-31T23:00:00Z").dropna(subset=["demand", "forecast", "residual"])
    rep = b1.split(f, "2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z").dropna(subset=["demand", "forecast", "residual"])
    if min(len(train), len(val), len(rep)) == 0:
        raise RuntimeError(f"Empty cohort train={len(train)} val={len(val)} rep={len(rep)}")

    frozen_b1_path = Path("experiments/pfsce_necf_confirmatory_2024/results/B1_2024.json")
    frozen = json.loads(frozen_b1_path.read_text())
    frozen_weights = np.asarray(frozen["stack_weights"], dtype=float)
    if len(frozen_weights) != 3 or not np.isclose(frozen_weights.sum(), 1.0, atol=1e-6):
        raise RuntimeError(f"Invalid frozen stack weights: {frozen_weights}")

    # Reproduce the validation-only model selection, but never let 2025 influence it.
    ridge, ridge_meta = b1.ridge_fit(train[F], train.residual, val[F], val.residual)
    if float(ridge_meta["alpha"]) != EXPECTED_ALPHA:
        raise RuntimeError(f"Frozen ridge alpha not reproduced: {ridge_meta}")
    hgb = b1.hgb_fit(train[F], train.residual)

    val_ridge = val.forecast.to_numpy() + ridge.predict(val[F])
    val_hgb = val.forecast.to_numpy() + hgb.predict(val[F])
    val_components = np.column_stack([val.forecast.to_numpy(), val_ridge, val_hgb])
    val = val.copy()
    val["pred_stack"] = val_components @ frozen_weights

    validation_router = {}
    weekly_val = b1.weekly_lifts(val, "pred_stack")
    for ba in b1.BAS:
        ci = b1.boot([v for b, _, v in weekly_val if b == ba])
        promoted = bool(ci["lower"] is not None and ci["lower"] > 0)
        validation_router[ba] = {
            "promoted": promoted,
            "validation_BA_week_relative_lift_bootstrap": ci,
        }
        if promoted != EXPECTED_PROMOTED[ba]:
            raise RuntimeError(
                f"Frozen router decision not reproduced for {ba}: got={promoted} expected={EXPECTED_PROMOTED[ba]} ci={ci}"
            )

    # Exact frozen router decisions and exact frozen stack weights are applied to 2025.
    rep_ridge = rep.forecast.to_numpy() + ridge.predict(rep[F])
    rep_hgb = rep.forecast.to_numpy() + hgb.predict(rep[F])
    rep_stack = np.column_stack([rep.forecast.to_numpy(), rep_ridge, rep_hgb]) @ frozen_weights
    rep = rep.copy()
    rep["pred_ridge"] = rep_ridge
    rep["pred_hgb"] = rep_hgb
    rep["pred_stack"] = rep_stack
    rep["router_promoted"] = rep.ba_code.map(EXPECTED_PROMOTED)
    rep["pred_routed"] = np.where(rep.router_promoted, rep.pred_stack, rep.forecast)

    models = {}
    for name, col in [
        ("operator", "forecast"),
        ("ridge", "pred_ridge"),
        ("hgb", "pred_hgb"),
        ("stack", "pred_stack"),
        ("routed", "pred_routed"),
    ]:
        models[name] = b1.metrics(rep.demand, rep[col], rep.forecast if name != "operator" else None)

    by_ba = {}
    promoted_positive = True
    fallback_unchanged = True
    for ba, g in rep.groupby("ba_code"):
        ba = str(ba)
        br = {
            "router_promoted": EXPECTED_PROMOTED[ba],
            "operator": b1.metrics(g.demand, g.forecast),
            "routed": b1.metrics(g.demand, g.pred_routed, g.forecast),
            "stack": b1.metrics(g.demand, g.pred_stack, g.forecast),
            "ridge": b1.metrics(g.demand, g.pred_ridge, g.forecast),
            "hgb": b1.metrics(g.demand, g.pred_hgb, g.forecast),
        }
        by_ba[ba] = br
        if EXPECTED_PROMOTED[ba]:
            promoted_positive = promoted_positive and br["routed"]["relative_mae_improvement"] > 0
        else:
            unchanged = np.allclose(g.pred_routed.to_numpy(), g.forecast.to_numpy(), rtol=0, atol=0)
            fallback_unchanged = fallback_unchanged and unchanged
            br["fallback_exactly_unchanged"] = bool(unchanged)

    cluster = weekly_cluster_bootstrap(rep, "pred_routed")
    routed_lift = float(models["routed"]["relative_mae_improvement"])
    conditions = {
        "aggregate_lift_ge_10pct": routed_lift >= 0.10,
        "BA_week_bootstrap_lower_gt_zero": cluster["lower"] > 0,
        "CISO_and_PJM_positive": bool(promoted_positive),
        "fallback_BAs_unchanged": bool(fallback_unchanged),
    }
    if all(conditions.values()):
        verdict = "STRONG_REPLICATION"
    elif conditions["BA_week_bootstrap_lower_gt_zero"] and conditions["CISO_and_PJM_positive"] and conditions["fallback_BAs_unchanged"] and routed_lift > 0:
        verdict = "DIRECTIONAL_REPLICATION"
    else:
        verdict = "REPLICATION_FAILURE"

    result = {
        "replication_id": REPLICATION_ID,
        "protocol_family": PROTOCOL_FAMILY,
        "status": "FROZEN_B1_2025_REPLICATION_RESULT",
        "authority": SOURCE_AUTHORITY,
        "frozen_2024_B1_result_sha256": hashlib.sha256(frozen_b1_path.read_bytes()).hexdigest(),
        "source_2025": target_provenance,
        "cohorts": {
            "n_train_2020_2022": int(len(train)),
            "n_validation_2023": int(len(val)),
            "n_replication_2025": int(len(rep)),
            "n_quarantined_combined": int(len(bad)),
        },
        "frozen_model": {
            "ridge_alpha": EXPECTED_ALPHA,
            "stack_weights_operator_ridge_hgb": frozen_weights.tolist(),
            "router_expected": EXPECTED_PROMOTED,
            "router_reproduction_2023": validation_router,
        },
        "models_2025": models,
        "by_ba_2025": by_ba,
        "BA_week_routed_lift_bootstrap": {k: v for k, v in cluster.items() if k != "clusters"},
        "replication_gate": {
            "required": {
                "aggregate_routed_MAE_lift_min": 0.10,
                "BA_week_bootstrap_lower_gt": 0.0,
                "promoted_BAs_positive": ["CISO", "PJM"],
                "fallback_BAs_unchanged": ["ERCO", "ISNE"],
            },
            "observed_routed_relative_mae_lift": routed_lift,
            "conditions": conditions,
            "verdict": verdict,
        },
    }

    result_path = out / "B1_2025_REPLICATION.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    rep.to_csv(out / "B1_2025_predictions.csv.gz", index=False, compression="gzip")
    bad.to_csv(out / "B1_2025_quarantined_combined.csv.gz", index=False, compression="gzip")
    (out / "B1_2025_source_provenance.json").write_text(json.dumps(target_provenance, indent=2, sort_keys=True))

    lines = [
        "# B1 2025 Replication Verdict",
        "",
        f"Replication: `{REPLICATION_ID}`",
        f"Authority: `{SOURCE_AUTHORITY}`",
        "",
        f"**Verdict: {verdict}.**",
        "",
        f"Operator MAE: **{models['operator']['mae']:.3f} MW**",
        f"Frozen routed B1 MAE: **{models['routed']['mae']:.3f} MW**",
        f"Aggregate routed MAE improvement: **{routed_lift*100:.3f}%**",
        f"BA-week bootstrap 95% CI: **[{cluster['lower']*100:.3f}%, {cluster['upper']*100:.3f}%]**",
        f"BA-week clusters: **{cluster['n_clusters']}**",
        "",
        "## Frozen replication gate",
        "",
        *[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in conditions.items()],
        "",
        "This replication used no 2024 or 2025 outcomes for fitting, hyperparameter selection, stack selection, or router selection. 2024 observations may enter only as prior-state lags for 2025 targets, which is temporally legitimate. The 2025 target archive is a post-period EIA/PUDL vintage, so the result remains revision-sensitive rather than highest-authority time-capsule evidence.",
    ]
    (out / "B1_2025_REPLICATION_VERDICT.md").write_text("\n".join(lines) + "\n")

    checksums = {}
    for p in sorted(out.iterdir()):
        if p.is_file():
            checksums[p.name] = digest(p, "sha256")
    (out / "B1_2025_SHA256.json").write_text(json.dumps(checksums, indent=2, sort_keys=True))

    print("PFSCE_B1_2025_REPLICATION_BEGIN")
    print(json.dumps(result, indent=2))
    print("PFSCE_B1_2025_REPLICATION_END")


if __name__ == "__main__":
    main()
