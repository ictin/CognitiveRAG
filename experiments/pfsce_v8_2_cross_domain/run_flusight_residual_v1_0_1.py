"""Implementation-only patch for PFSCE-V8.2-FLUSIGHT-RESIDUAL-V1.

The first execution stopped before holdout scoring because pandas represented numeric
horizons as floats (1.0/2.0/3.0), while the frozen router keys are strings
"1"/"2"/"3". No forecast model, feature, cohort, threshold, metric, router rule,
or scientific interpretation is changed here. We normalize horizon identity to int
inside the existing attach_correction output and then execute the frozen v1 main.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_flusight_residual_v1 as frozen

_original_attach = frozen.attach_correction


def _attach_with_canonical_horizon(qf, cases, corr, col="correction"):
    out = _original_attach(qf, cases, corr, col=col)
    out["horizon"] = out["horizon"].astype(int)
    return out


frozen.attach_correction = _attach_with_canonical_horizon


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flusight-repo", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("experiment_output/flusight_v1"))
    args = ap.parse_args()
    frozen.main(args.flusight_repo, args.out_dir)
