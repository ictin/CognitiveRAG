from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def normalize_web_result(
    *,
    raw: Dict[str, Any],
    query: str,
    query_variant: str,
    rank: int,
    fetched_at: str | None = None,
) -> Dict[str, Any]:
    fetched = fetched_at or _iso_utc_now()
    title = raw.get("title") or raw.get("name") or ""
    url = raw.get("url") or raw.get("href") or raw.get("source_url") or ""
    snippet = raw.get("snippet") or raw.get("body") or raw.get("summary") or ""
    extracted = raw.get("text") or raw.get("content") or snippet
    published_at = raw.get("published_at") or raw.get("date") or None
    updated_at = raw.get("updated_at") or None

    trust_score = 0.6
    if isinstance(url, str) and ("wikipedia.org" in url or "docs." in url or "developer." in url):
        trust_score = 0.8
    elif isinstance(url, str) and url.startswith("https://"):
        trust_score = 0.7

    source_id = url or f"title:{title.lower().strip()}"
    content_hash = _content_hash(extracted)
    freshness_class = "warm"
    q = (query or "").lower()
    if any(t in q for t in ("latest", "current", "today", "news", "update")):
        freshness_class = "hot"

    return {
        "query": query,
        "query_variant": query_variant,
        "rank": int(rank),
        "title": title,
        "url": url,
        "source_id": source_id,
        "snippet": snippet,
        "extracted_text": extracted,
        "fetched_at": fetched,
        "published_at": published_at,
        "updated_at": updated_at,
        "trust_score": trust_score,
        "freshness_class": freshness_class,
        "content_hash": content_hash,
        "raw": raw,
    }
