from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_b1 as b1
import run_eia_revision_audit as audit

CORRECTED_AUDIT_ID = "NECF-001-EIA930-REVISION-AUDIT-v1.0.1"
EXPECTED_ORIGINAL = {
    "n_train": 52538,
    "n_validation": 17506,
    "n_test": 17483,
    "operator_mae": 1725.0005719842131,
}


def corrected_load_record_years(record_id: str, years: list[int], cache: Path):
    """Assemble complete archive files first, then UTC-filter the combined stream.

    v1 incorrectly UTC-filtered each nominal half/year archive before concatenation.
    EIA half-year files are organized by reporting periods/local boundaries, so a
    few UTC boundary hours can live in the adjacent nominal archive. Filtering
    each archive first removed those rows and changed the frozen cohort by a few
    observations. This correction is mechanical and does not change any model,
    threshold, router, or interpretation rule.
    """
    years = sorted(years)
    filenames: list[str] = []
    # For a one-year row-level comparison include the previous H2 so that UTC
    # hours crossing the local calendar boundary are represented exactly as they
    # are when the continuous archive sequence is assembled.
    if len(years) == 1:
        filenames.append(f"eia930-{years[0]-1}half2.zip")
    for year in years:
        filenames.extend(audit.half_filenames(year))

    frames = []
    provenance = []
    for fn in filenames:
        print("DOWNLOAD_CORRECTED", record_id, fn, flush=True)
        p, prov = audit.download_record_file(record_id, fn, cache)
        d, members = b1.read_zip(p)
        frames.append(d)
        provenance.append({**prov, "raw_member_rows": int(len(d)), "members": members})

    x = pd.concat(frames, ignore_index=True)
    x = x.sort_values(["ba", "datetime_utc"]).drop_duplicates(["ba", "datetime_utc"], keep="last")
    lo = pd.Timestamp(f"{years[0]}-01-01T00:00:00Z")
    hi = pd.Timestamp(f"{years[-1]}-12-31T23:59:59Z")
    x = x[(x.datetime_utc >= lo) & (x.datetime_utc <= hi)].copy()
    return x, provenance


def main() -> None:
    audit.AUDIT_ID = CORRECTED_AUDIT_ID
    audit.load_record_years = corrected_load_record_years
    audit.main()

    out = Path("experiment_output/eia_revision_audit")
    rp = out / "EIA930_REVISION_AUDIT.json"
    result = json.loads(rp.read_text())
    original = result["full_pipeline_2020_2024"]["results"]["14881638"]

    checks = {
        "n_train_exact": original["n_train"] == EXPECTED_ORIGINAL["n_train"],
        "n_validation_exact": original["n_validation"] == EXPECTED_ORIGINAL["n_validation"],
        "n_test_exact": original["n_test"] == EXPECTED_ORIGINAL["n_test"],
        "operator_mae_exact": abs(original["operator"]["mae"] - EXPECTED_ORIGINAL["operator_mae"]) < 1e-9,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Corrected audit does not reproduce frozen source cohort: {checks}, observed={original}")

    result["implementation_correction"] = {
        "version": "v1.0.1",
        "reason": "v1 filtered each nominal archive to its UTC calendar year before concatenation, dropping UTC boundary rows stored in adjacent reporting archives",
        "scope": "source assembly only; no model, gate, router, threshold, or interpretation changed",
        "frozen_original_cohort_reproduction": checks,
    }
    rp.write_text(json.dumps(result, indent=2, sort_keys=True))

    verdict = out / "EIA930_REVISION_AUDIT_VERDICT.md"
    txt = verdict.read_text()
    txt = txt.replace(
        "# EIA-930 Revision Sensitivity Audit\n",
        "# EIA-930 Revision Sensitivity Audit\n\nImplementation correction: `v1.0.1` — exact continuous archive assembly before UTC filtering. Frozen original cohort counts and operator MAE reproduced exactly.\n",
        1,
    )
    verdict.write_text(txt)
    print("PFSCE_EIA_REVISION_AUDIT_V1_0_1_REPRODUCTION", json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
