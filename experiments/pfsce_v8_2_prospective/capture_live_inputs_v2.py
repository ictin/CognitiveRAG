from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import requests

import capture_live_inputs as base

HISTORY_DAYS = 35
PAGE_SIZE = 5000


def _sanitize(obj):
    """Remove credentials echoed by the EIA API before anything is persisted."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() == "api_key":
                out[k] = "[REDACTED]"
            else:
                out[k] = _sanitize(v)
        return out
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def eia_capture(api_key: str | None, now_utc: datetime, cutoff_utc: datetime, out):
    result = {
        "source": "EIA API v2 electricity/rto/region-data",
        "retrieval_time_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "respondents": base.BAS,
        "types": base.TYPES,
        "history_days_requested": HISTORY_DAYS,
        "eligible": False,
        "credential_redaction": "EIA request metadata api_key removed before persistence",
    }
    if not api_key:
        result["status"] = "EIA_API_KEY_MISSING"
        base.dump(out / "eia_capture_status.json", result)
        return result

    sess = requests.Session()
    sess.headers.update({"User-Agent": "PFSCE-v8.2-prospective-validation/1.2"})
    endpoint = "https://api.eia.gov/v2/electricity/rto/region-data"
    try:
        meta_r = sess.get(endpoint, params={"api_key": api_key}, timeout=45)
        meta_r.raise_for_status()
        meta_obj = _sanitize(meta_r.json())
        meta_raw = _json_bytes(meta_obj)
        (out / "eia_route_metadata.raw.json").write_bytes(meta_raw)
        result["metadata_sha256"] = base.sha256_bytes(meta_raw)
        result["metadata_http_status"] = meta_r.status_code

        local = now_utc.astimezone(base.ET)
        start = (local.date() - timedelta(days=HISTORY_DAYS)).isoformat() + "T00"
        end = (local.date() + timedelta(days=1)).isoformat() + "T23"
        common: list[tuple[str, str]] = [
            ("api_key", api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            ("start", start),
            ("end", end),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "desc"),
        ]
        for ba in base.BAS:
            common.append(("facets[respondent][]", ba))
        for typ in base.TYPES:
            common.append(("facets[type][]", typ))

        rows = []
        page_meta = []
        offset = 0
        first_response = None
        while True:
            params = list(common) + [("offset", str(offset)), ("length", str(PAGE_SIZE))]
            r = sess.get(endpoint + "/data/", params=params, timeout=60)
            r.raise_for_status()
            obj = r.json()
            if first_response is None:
                first_response = _sanitize(obj)
            page = obj.get("response", {}).get("data", [])
            rows.extend(page)
            page_meta.append({"offset": offset, "rows": len(page), "http_status": r.status_code})
            total = int(obj.get("response", {}).get("total", len(rows)))
            if not page or len(rows) >= total:
                break
            offset += len(page)
            if offset > 50000:
                raise RuntimeError("unexpected EIA pagination size")

        persisted = _sanitize(first_response or {})
        persisted.setdefault("response", {})["data"] = rows
        persisted["capture_provenance"] = {
            "history_start": start,
            "history_end": end,
            "pages": page_meta,
            "credential_redacted_before_persistence": True,
            "raw_http_bytes_with_credential_not_persisted": True,
        }
        raw = _json_bytes(persisted)
        (out / "eia_region_data.raw.json").write_bytes(raw)
        result["payload_sha256"] = base.sha256_bytes(raw)
        result["payload_http_status"] = 200
        result["row_count"] = len(rows)
        result["history_start"] = start
        result["history_end"] = end
        result["pages"] = page_meta

        seen_ba = sorted({str(r.get("respondent")) for r in rows if r.get("respondent") is not None})
        seen_type = sorted({str(r.get("type")) for r in rows if r.get("type") is not None})
        periods = sorted(str(r.get("period")) for r in rows if r.get("period"))
        result["seen_respondents"] = seen_ba
        result["seen_types"] = seen_type
        result["earliest_period"] = periods[0] if periods else None
        result["latest_period"] = periods[-1] if periods else None

        required_history_date = local.date() - timedelta(days=33)
        history_ok = bool(periods) and periods[0][:10] <= required_history_date.isoformat()
        result["history_coverage_ok_for_b1_features"] = history_ok
        result["eligible"] = (
            bool(rows)
            and all(x in seen_ba for x in base.BAS)
            and all(x in seen_type for x in base.TYPES)
            and history_ok
            and datetime.now(timezone.utc) <= cutoff_utc
        )
        result["status"] = "CAPTURED_PRE_CUTOFF" if result["eligible"] else "CAPTURED_BUT_INCOMPLETE_OR_LATE"
    except Exception as exc:
        result["status"] = "EIA_CAPTURE_FAILED"
        result["error"] = repr(exc)
    base.dump(out / "eia_capture_status.json", result)
    return result


# Transport-only override: target definitions, model rules, HRRR gate, and promotion gates remain frozen.
base.eia_capture = eia_capture

if __name__ == "__main__":
    raise SystemExit(base.main())
