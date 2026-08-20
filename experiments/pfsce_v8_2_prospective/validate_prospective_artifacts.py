from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fail(msg: str) -> None:
    raise AssertionError(msg)


def probability(x, label: str) -> None:
    if x is None:
        return
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)) or not (0.0 <= float(x) <= 1.0):
        fail(f"{label}: invalid probability {x!r}")


def validate_general(path: Path) -> int:
    d = json.loads(path.read_text(encoding="utf-8"))
    for k in ("cohort_id", "issue_time", "methodology", "packets", "nominal_n"):
        if k not in d:
            fail(f"{path}: missing {k}")
    packets = d["packets"]
    if d["nominal_n"] != len(packets):
        fail(f"{path}: nominal_n != packet count")
    ids = []
    for p in packets:
        for k in ("id", "topic", "baseline_probability", "pfsce_raw_probability", "authority", "resolution_date"):
            if k not in p:
                fail(f"{path}: packet missing {k}: {p}")
        ids.append(p["id"])
        probability(p["baseline_probability"], f"{path}:{p['id']}:baseline")
        probability(p["pfsce_raw_probability"], f"{path}:{p['id']}:pfsce")
    if len(ids) != len(set(ids)):
        fail(f"{path}: duplicate packet IDs")
    return len(packets)


def validate_cycle(path: Path) -> int:
    d = json.loads(path.read_text(encoding="utf-8"))
    for k in ("cycle_id", "contract_freeze_commit", "issue_date", "methodology", "anchor_generation", "challenger_generation", "denominator", "packets"):
        if k not in d:
            fail(f"{path}: missing {k}")
    packets = d["packets"]
    if d["denominator"].get("nominal_n") != len(packets):
        fail(f"{path}: denominator nominal_n != packet count")
    ids = []
    required = (
        "forecast_id", "stream", "question", "resolution_contract_ref", "forecastability_cell",
        "mode", "baseline", "anchor_raw_probability", "anchor_routed_probability",
        "anchor_router_action", "probability_provenance", "calibration_status",
        "authority_grade", "meg", "evidence_lineages", "effective_evidence_n",
        "update_triggers", "prior_trajectory", "episode_cluster",
        "exposure_intervention_status", "scoring_status",
    )
    for p in packets:
        for k in required:
            if k not in p:
                fail(f"{path}: {p.get('forecast_id','?')} missing {k}")
        fid = p["forecast_id"]
        ids.append(fid)
        b = p["baseline"]
        if not isinstance(b, dict) or "probability" not in b or "availability_time" not in b or "source" not in b:
            fail(f"{path}:{fid}: incomplete baseline provenance")
        probability(b.get("probability"), f"{path}:{fid}:baseline")
        probability(p.get("anchor_raw_probability"), f"{path}:{fid}:anchor_raw")
        probability(p.get("anchor_routed_probability"), f"{path}:{fid}:anchor_routed")
        probability(p.get("challenger_raw_probability"), f"{path}:{fid}:challenger")
        probability(p.get("calibrated_probability"), f"{path}:{fid}:calibrated")
        if not p["evidence_lineages"]:
            fail(f"{path}:{fid}: evidence_lineages empty")
        for e in p["evidence_lineages"]:
            for k in ("lineage_id", "origin", "availability_time", "independence_weight"):
                if k not in e:
                    fail(f"{path}:{fid}: incomplete evidence lineage")
        if p["scoring_status"] not in {"UNRESOLVED", "RESOLVED", "AMBIGUOUS", "VOID_BY_CONTRACT"}:
            fail(f"{path}:{fid}: invalid scoring_status {p['scoring_status']}")
        if p["scoring_status"] == "RESOLVED":
            if "resolution" not in p or "resolution_time" not in p or "resolution_source" not in p:
                fail(f"{path}:{fid}: resolved without resolution provenance")
    if len(ids) != len(set(ids)):
        fail(f"{path}: duplicate forecast IDs")
    counts = d["denominator"].get("anchor_router_counts", {})
    if counts:
        if sum(int(v) for v in counts.values()) != len(packets):
            fail(f"{path}: router counts do not preserve denominator")
    return len(packets)


def validate_necf_day(day: Path) -> int:
    # Once live NECF directories exist, enforce the frozen minimum artifact contract.
    required = ["issue_manifest.json", "hrrr_availability.json", "integrity.json"]
    for name in required:
        if not (day / name).exists():
            fail(f"{day}: missing required live artifact {name}")
    if not ((day / "eia_issue_snapshot.json").exists() or (day / "eia_issue_snapshot.sha256").exists()):
        fail(f"{day}: missing EIA issue snapshot/hash")
    if not ((day / "predictions.json").exists() or (day / "predictions.csv").exists()):
        fail(f"{day}: missing predictions artifact")
    manifest = json.loads((day / "issue_manifest.json").read_text(encoding="utf-8"))
    for k in ("protocol_id", "forecast_origin", "target_end_timestamps", "router"):
        if k not in manifest:
            fail(f"{day}: issue_manifest missing {k}")
    if len(manifest["target_end_timestamps"]) != 12:
        fail(f"{day}: expected exactly 12 target hours")
    integ = json.loads((day / "integrity.json").read_text(encoding="utf-8"))
    if integ.get("verdict") not in {"INTEGRITY_PASS", "INTEGRITY_PASS_WITH_QUARANTINE", "INTEGRITY_FAIL", "PENDING"}:
        fail(f"{day}: invalid integrity verdict")
    return 1


def main() -> int:
    n = 0
    general = ROOT / "GENERAL_COHORT_20260820.json"
    if general.exists():
        n += validate_general(general)

    cycle_files = sorted((ROOT / "cycles").glob("*-forecast-ledger.json")) if (ROOT / "cycles").exists() else []
    for p in cycle_files:
        n += validate_cycle(p)

    live = ROOT / "necf_live"
    if live.exists():
        for d in sorted(x for x in live.iterdir() if x.is_dir()):
            n += validate_necf_day(d)

    if n == 0:
        fail("no prospective artifacts found")
    print(f"PFSCE prospective integrity validation PASS: {n} packet/day objects checked")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PFSCE prospective integrity validation FAIL: {exc}", file=sys.stderr)
        raise
