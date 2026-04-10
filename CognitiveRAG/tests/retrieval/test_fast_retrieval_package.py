import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.retrieval import fast_lanes
from CognitiveRAG.crag.retrieval.router import AgentHotCache, clear_hot_cache, get_hot_cache_stats, route_and_retrieve
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def _seed_promoted(
    tmp_path: Path,
    *,
    promoted_id: str,
    fact: str,
    tier: str,
    state: str = WebPromotedMemoryStore.STATE_TRUSTED,
    lifecycle: str = WebPromotedMemoryStore.FRESHNESS_FRESH,
    confidence: float = 0.8,
    evidence_count: int = 2,
    metadata: dict | None = None,
    last_validated_at: str = "2026-04-03T09:00:00Z",
) -> None:
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id=promoted_id,
        canonical_fact=fact,
        evidence_ids=[f"ev-{promoted_id}-{i}" for i in range(evidence_count)],
        confidence=confidence,
        freshness_state="warm",
        metadata={"source_url": f"https://example.com/{promoted_id}", **(metadata or {})},
        now_iso="2026-04-03T10:00:00Z",
        promotion_state=state,
        promotion_tier=tier,
        origin_tier=WebPromotedMemoryStore.TIER_LOCAL,
        freshness_lifecycle_state=lifecycle,
        freshness_reason="seed",
        last_validated_at=last_validated_at,
    )


def test_agent_hot_cache_hits_on_repeated_query(tmp_path: Path):
    clear_hot_cache()
    _seed_promoted(
        tmp_path,
        promoted_id="wp_fast_1",
        fact="postgres rollout steps are guarded by release policy.",
        tier=WebPromotedMemoryStore.TIER_WORKSPACE,
    )

    plan1, hits1 = route_and_retrieve(
        query="postgres rollout steps",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="s-cache",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=4,
    )
    assert "agent_hot_cache_hit" not in plan1.reason
    assert hits1
    assert all(not bool((h.provenance or {}).get("fast_path", {}).get("agent_hot_cache_hit")) for h in hits1)

    plan2, hits2 = route_and_retrieve(
        query="postgres rollout steps",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="s-cache",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=4,
    )
    assert "agent_hot_cache_hit" in plan2.reason
    assert hits2
    assert all(bool((h.provenance or {}).get("fast_path", {}).get("agent_hot_cache_hit")) for h in hits2)

    stats = get_hot_cache_stats()
    assert int(stats["hits"]) >= 1
    assert int(stats["misses"]) >= 1


def test_agent_hot_cache_misses_when_query_changes(tmp_path: Path):
    clear_hot_cache()
    _seed_promoted(
        tmp_path,
        promoted_id="wp_fast_2",
        fact="Cache keys should track query normalization deterministically.",
        tier=WebPromotedMemoryStore.TIER_LOCAL,
    )

    route_and_retrieve(
        query="cache key behavior",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="s-miss",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=4,
    )

    plan2, _ = route_and_retrieve(
        query="cache key behavior changed",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="s-miss",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=4,
    )
    assert "agent_hot_cache_hit" not in plan2.reason


def test_hot_cache_bounded_size_and_expiry():
    cache = AgentHotCache(max_entries=2, ttl_seconds=0.01)
    from CognitiveRAG.crag.retrieval.router import RoutePlan

    plan = RoutePlan(
        intent_family=IntentFamily.MEMORY_SUMMARY,
        lanes=[RetrievalLane.PROMOTED],
        reason="x",
    )
    cache.set(("a",), plan, [])
    cache.set(("b",), plan, [])
    cache.set(("c",), plan, [])
    assert int(cache.stats()["entries"]) == 2
    assert int(cache.stats()["evictions"]) >= 1

    time.sleep(0.02)
    assert cache.get(("b",)) is None
    assert int(cache.stats()["misses"]) >= 1


def test_fast_lanes_distinguish_installation_workspace_and_global(tmp_path: Path):
    _seed_promoted(
        tmp_path,
        promoted_id="wp_local",
        fact="Local lane note about retention hooks.",
        tier=WebPromotedMemoryStore.TIER_LOCAL,
    )
    _seed_promoted(
        tmp_path,
        promoted_id="wp_workspace",
        fact="Workspace lane note about retention hooks.",
        tier=WebPromotedMemoryStore.TIER_WORKSPACE,
    )
    _seed_promoted(
        tmp_path,
        promoted_id="wp_global",
        fact="Global lane note about retention hooks.",
        tier=WebPromotedMemoryStore.TIER_GLOBAL,
    )

    hits = fast_lanes.retrieve_fast_lanes(
        workdir=str(tmp_path),
        query="retention hooks",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        top_k=6,
    )
    lanes = {h.lane for h in hits}
    assert RetrievalLane.INSTALLATION_FAST in lanes
    assert RetrievalLane.WORKSPACE_FAST in lanes
    assert RetrievalLane.GLOBAL_PROMOTED in lanes

    tier_by_lane = {h.lane: str((h.provenance or {}).get("promotion_tier") or "") for h in hits}
    assert tier_by_lane[RetrievalLane.INSTALLATION_FAST] == "local"
    assert tier_by_lane[RetrievalLane.WORKSPACE_FAST] == "workspace"
    assert tier_by_lane[RetrievalLane.GLOBAL_PROMOTED] == "global"


def test_fast_lane_metadata_preserves_freshness_and_contradiction_truth(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    now = datetime.now(timezone.utc)
    fresh_validated_at = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    stale_validated_at = (now - timedelta(hours=200)).isoformat().replace("+00:00", "Z")

    _seed_promoted(
        tmp_path,
        promoted_id="wp_safe",
        fact="Feature flag is enabled in production.",
        tier=WebPromotedMemoryStore.TIER_GLOBAL,
        lifecycle=WebPromotedMemoryStore.FRESHNESS_FRESH,
        metadata={"claim_key": "rollout_status", "claim_value": "enabled", "source_class": "web_promoted"},
        last_validated_at=fresh_validated_at,
    )
    _seed_promoted(
        tmp_path,
        promoted_id="wp_conflict_a",
        fact="Feature flag is enabled in production.",
        tier=WebPromotedMemoryStore.TIER_GLOBAL,
        lifecycle=WebPromotedMemoryStore.FRESHNESS_STALE,
        metadata={"claim_key": "feature_flag_enabled", "claim_value": "true", "source_class": "local_durable"},
        last_validated_at=stale_validated_at,
    )
    _seed_promoted(
        tmp_path,
        promoted_id="wp_conflict_b",
        fact="Feature flag is not enabled in production.",
        tier=WebPromotedMemoryStore.TIER_GLOBAL,
        lifecycle=WebPromotedMemoryStore.FRESHNESS_STALE,
        metadata={"claim_key": "feature_flag_enabled", "claim_value": "false", "source_class": "web_promoted"},
        last_validated_at=stale_validated_at,
    )

    # Contradiction is recorded by deterministic claim key/value mismatch.
    summary = store.get_contradiction_summary("wp_conflict_a")
    assert summary["has_contradiction"] is True

    hits = fast_lanes.retrieve_fast_lanes(
        workdir=str(tmp_path),
        query="feature flag",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=6,
    )
    by_id = {h.id: h for h in hits}
    safe = by_id["global_promoted:wp_safe"]
    conflict = by_id["global_promoted:wp_conflict_a"]

    assert safe.provenance.get("freshness_lifecycle_state") == "fresh"
    assert conflict.provenance.get("freshness_lifecycle_state") == "stale"
    assert bool(conflict.provenance.get("contradiction", {}).get("has_contradiction")) is True
    assert bool(safe.provenance.get("fast_lane", {}).get("cache_safe")) is True
    assert bool(conflict.provenance.get("fast_lane", {}).get("cache_safe")) is False
    assert float(conflict.trust_score) <= float(safe.trust_score)


def test_route_and_retrieve_falls_back_when_fast_lanes_empty(tmp_path: Path):
    clear_hot_cache()
    plan, hits = route_and_retrieve(
        query="what did we decide in this session",
        intent_family=IntentFamily.EXACT_RECALL,
        session_id="s-fallback",
        fresh_tail=[{"index": 1, "text": "we agreed to stage rollout", "message_id": "m1"}],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=4,
    )
    assert RetrievalLane.EPISODIC in plan.lanes
    assert any(h.lane in {RetrievalLane.FRESH_TAIL, RetrievalLane.EPISODIC, RetrievalLane.SESSION_SUMMARY} for h in hits)
    assert all(h.lane not in {RetrievalLane.INSTALLATION_FAST, RetrievalLane.WORKSPACE_FAST, RetrievalLane.GLOBAL_PROMOTED} for h in hits)


def test_fast_lane_order_deterministic_for_repeated_runs(tmp_path: Path):
    _seed_promoted(
        tmp_path,
        promoted_id="wp_order_a",
        fact="Workspace promotion A for deterministic order.",
        tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        confidence=0.81,
    )
    _seed_promoted(
        tmp_path,
        promoted_id="wp_order_b",
        fact="Workspace promotion B for deterministic order.",
        tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        confidence=0.79,
    )

    first = [
        h.id
        for h in fast_lanes.retrieve_fast_lanes(
            workdir=str(tmp_path),
            query="deterministic order",
            intent_family=IntentFamily.PLANNING,
            top_k=6,
        )
    ]
    second = [
        h.id
        for h in fast_lanes.retrieve_fast_lanes(
            workdir=str(tmp_path),
            query="deterministic order",
            intent_family=IntentFamily.PLANNING,
            top_k=6,
        )
    ]
    assert first == second
