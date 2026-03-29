from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Sequence

from .evidence_store import WebEvidenceStore
from .fetch_log import WebFetchLogStore
from .normalize import normalize_web_result
from .query_planner import WebNeedDecision, WebQueryPlan


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class WebFetcher:
    """Fetch + normalize + persist web evidence with cache reuse."""

    def __init__(
        self,
        *,
        evidence_store: WebEvidenceStore,
        fetch_log: WebFetchLogStore,
        provider: Callable[[str, int], Sequence[Dict[str, Any]]] | None = None,
    ):
        self.evidence_store = evidence_store
        self.fetch_log = fetch_log
        self.provider = provider or self._default_provider

    def _default_provider(self, query: str, top_k: int) -> Sequence[Dict[str, Any]]:
        try:
            from ddgs import DDGS  # type: ignore

            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=top_k))
        except Exception:
            try:
                from duckduckgo_search import DDGS  # type: ignore

                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=top_k))
            except Exception:
                return []

    def fetch_plan(
        self,
        *,
        plan: WebQueryPlan,
        need: WebNeedDecision,
        min_cache_hits: int = 2,
    ) -> List[Dict[str, Any]]:
        cached = self.evidence_store.search(plan.query, top_k=max(plan.max_results, min_cache_hits))
        if cached and (not need.freshness_sensitive or len(cached) >= min_cache_hits):
            self.fetch_log.append(
                query=plan.query,
                query_variant=plan.query,
                status="cache_hit",
                http_status=None,
                error=None,
                result_count=len(cached),
                fetched_at=_utc_now_iso(),
            )
            return cached[: plan.max_results]

        normalized_all: List[Dict[str, Any]] = []
        for variant in plan.variants:
            fetched_at = _utc_now_iso()
            try:
                raw_items = list(self.provider(variant, plan.max_results))
                for i, raw in enumerate(raw_items):
                    norm = normalize_web_result(
                        raw=raw,
                        query=plan.query,
                        query_variant=variant,
                        rank=i,
                        fetched_at=fetched_at,
                    )
                    evidence_id = self.evidence_store.upsert_evidence(norm)
                    norm["evidence_id"] = evidence_id
                    normalized_all.append(norm)
                self.fetch_log.append(
                    query=plan.query,
                    query_variant=variant,
                    status="ok",
                    http_status=200,
                    error=None,
                    result_count=len(raw_items),
                    fetched_at=fetched_at,
                )
            except Exception as exc:
                self.fetch_log.append(
                    query=plan.query,
                    query_variant=variant,
                    status="error",
                    http_status=None,
                    error=str(exc),
                    result_count=0,
                    fetched_at=fetched_at,
                )
        if not normalized_all:
            return cached[: plan.max_results]
        return normalized_all[: plan.max_results]
