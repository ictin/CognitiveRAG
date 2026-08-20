from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config

PROTOCOL = "PFSCE-V8.2-NECF-PROSPECTIVE-20260820-v1"
ET = ZoneInfo("America/New_York")
BAS = ["CISO", "ERCO", "PJM", "ISNE"]
TYPES = ["D", "DF"]
BUCKET = "noaa-hrrr-bdp-pds"


def canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def eia_capture(api_key: str | None, now_utc: datetime, cutoff_utc: datetime, out: Path) -> dict:
    result = {
        "source": "EIA API v2 electricity/rto/region-data",
        "retrieval_time_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "respondents": BAS,
        "types": TYPES,
        "eligible": False,
    }
    if not api_key:
        result["status"] = "EIA_API_KEY_MISSING"
        dump(out / "eia_capture_status.json", result)
        return result

    sess = requests.Session()
    sess.headers.update({"User-Agent": "PFSCE-v8.2-prospective-validation/1.0"})
    base = "https://api.eia.gov/v2/electricity/rto/region-data"
    try:
        meta_r = sess.get(base, params={"api_key": api_key}, timeout=45)
        meta_r.raise_for_status()
        meta_raw = meta_r.content
        (out / "eia_route_metadata.raw.json").write_bytes(meta_raw)
        result["metadata_sha256"] = sha256_bytes(meta_raw)
        result["metadata_http_status"] = meta_r.status_code

        local = now_utc.astimezone(ET)
        start = (local.date() - timedelta(days=1)).isoformat() + "T00"
        end = (local.date() + timedelta(days=1)).isoformat() + "T23"
        params: list[tuple[str, str]] = [
            ("api_key", api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            ("start", start),
            ("end", end),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "desc"),
            ("offset", "0"),
            ("length", "5000"),
        ]
        for ba in BAS:
            params.append(("facets[respondent][]", ba))
        for typ in TYPES:
            params.append(("facets[type][]", typ))

        data_r = sess.get(base + "/data/", params=params, timeout=60)
        data_r.raise_for_status()
        raw = data_r.content
        (out / "eia_region_data.raw.json").write_bytes(raw)
        result["payload_sha256"] = sha256_bytes(raw)
        result["payload_http_status"] = data_r.status_code
        payload = data_r.json()
        rows = payload.get("response", {}).get("data", [])
        result["row_count"] = len(rows)
        seen_ba = sorted({str(r.get("respondent")) for r in rows if r.get("respondent") is not None})
        seen_type = sorted({str(r.get("type")) for r in rows if r.get("type") is not None})
        result["seen_respondents"] = seen_ba
        result["seen_types"] = seen_type
        result["eligible"] = bool(rows) and all(x in seen_ba for x in BAS) and all(x in seen_type for x in TYPES) and now_utc <= cutoff_utc
        result["status"] = "CAPTURED_PRE_CUTOFF" if result["eligible"] else "CAPTURED_BUT_INCOMPLETE_OR_LATE"
    except Exception as exc:
        result["status"] = "EIA_CAPTURE_FAILED"
        result["error"] = repr(exc)
    dump(out / "eia_capture_status.json", result)
    return result


def hrrr_capture(issue_date: str, now_utc: datetime, cutoff_utc: datetime, out: Path) -> dict:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
    objects = []
    ok = True
    for lead in range(3, 15):
        key = f"hrrr.{issue_date.replace('-', '')}/conus/hrrr.t12z.wrfsfcf{lead:02d}.grib2"
        row = {"lead": lead, "key": key, "exists": False, "eligible_by_cutoff": False}
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            lm = head["LastModified"].astimezone(timezone.utc)
            row.update({
                "exists": True,
                "last_modified_utc": lm.isoformat().replace("+00:00", "Z"),
                "content_length": int(head.get("ContentLength", 0)),
                "etag": str(head.get("ETag", "")).strip('"'),
                "eligible_by_cutoff": lm <= cutoff_utc,
            })
        except Exception as exc:
            row["error"] = repr(exc)
        ok = ok and row["exists"] and row["eligible_by_cutoff"]
        objects.append(row)
    result = {
        "source": "NOAA NODD noaa-hrrr-bdp-pds",
        "bucket": BUCKET,
        "cycle": f"{issue_date} 12Z",
        "retrieval_time_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "objects": objects,
        "all_required_objects_eligible": bool(ok and now_utc <= cutoff_utc),
        "status": "HRRR_AVAILABILITY_GATE_PASS" if ok and now_utc <= cutoff_utc else "HRRR_AVAILABILITY_GATE_FAIL",
    }
    dump(out / "hrrr_availability_precutoff.json", result)
    return result


def main() -> int:
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)

    # The workflow is scheduled twice in UTC to survive US daylight-saving changes.
    # Exactly one invocation should land at 10:15 ET; all others exit without producing evidence.
    if not (now_et.hour == 10 and 8 <= now_et.minute <= 24):
        print(f"SKIP: local time {now_et.isoformat()} is outside the frozen pre-cutoff acquisition window")
        return 0

    cutoff_et = now_et.replace(hour=10, minute=30, second=0, microsecond=0)
    cutoff_utc = cutoff_et.astimezone(timezone.utc)
    issue_date = now_et.date().isoformat()
    out = Path("experiments/pfsce_v8_2_prospective/necf_precutoff") / issue_date
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol_id": PROTOCOL,
        "issue_date": issue_date,
        "capture_started_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "capture_started_et": now_et.isoformat(),
        "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "purpose": "transport-only pre-cutoff capture; no model or gate changes",
    }
    dump(out / "capture_manifest.json", manifest)

    eia = eia_capture(os.getenv("EIA_API_KEY"), now_utc, cutoff_utc, out)
    hrrr = hrrr_capture(issue_date, datetime.now(timezone.utc), cutoff_utc, out)

    summary = {
        **manifest,
        "eia_status": eia.get("status"),
        "eia_eligible": bool(eia.get("eligible", False)),
        "hrrr_status": hrrr.get("status"),
        "hrrr_eligible": bool(hrrr.get("all_required_objects_eligible", False)),
        "capture_promotion_eligible": bool(eia.get("eligible", False)),
        "note": "HRRR is optional for B1 A0/C1 and required only for W1; EIA baseline capture is required for B1 skill scoring.",
    }
    dump(out / "capture_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
