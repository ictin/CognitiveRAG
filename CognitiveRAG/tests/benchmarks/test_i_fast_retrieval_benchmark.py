from pathlib import Path

from CognitiveRAG.crag.benchmarks.i_fast_retrieval import run_fast_retrieval_benchmark, save_fast_retrieval_benchmark
from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def _seed(tmp_path: Path) -> None:
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_bench",
        canonical_fact="Workspace benchmark fact for repeated retrieval speed checks.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.86,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/bench", "source_class": "web_promoted"},
        now_iso="2026-04-03T10:00:00Z",
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        promotion_tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        origin_tier=WebPromotedMemoryStore.TIER_LOCAL,
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_FRESH,
        last_validated_at="2026-04-03T09:00:00Z",
    )


def test_fast_retrieval_benchmark_has_latency_and_cache_shape(tmp_path: Path):
    _seed(tmp_path)
    payload = run_fast_retrieval_benchmark(
        workdir=str(tmp_path),
        query="benchmark retrieval speed",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        repeats=3,
        top_k_per_lane=6,
    )

    assert payload["benchmark_type"] == "fast_retrieval_latency"
    assert payload["repeat_count"] == 3
    assert payload["latency"]["avg_ms"] >= 0
    assert len(payload["latency"]["runs_ms"]) == 3
    assert payload["cache"]["run_hits"] >= 1
    assert "category_routing" in payload
    assert "pruned_hit_count_total" in payload["category_routing"]
    assert payload["cache"]["run_misses"] >= 1
    assert "router_hot_cache" in payload["cache"]
    assert "route_cache" in payload["cache"]
    assert "topic_shortlist_cache" in payload["cache"]
    assert "rerank" in payload
    assert "applied_runs" in payload["rerank"]
    assert payload["runs"][0]["cache_hit"] is False
    assert "rerank" in payload["runs"][0]


def test_fast_retrieval_benchmark_can_save_json(tmp_path: Path):
    _seed(tmp_path)
    payload = run_fast_retrieval_benchmark(
        workdir=str(tmp_path),
        query="benchmark retrieval save",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        repeats=2,
    )
    out = save_fast_retrieval_benchmark(payload, output_dir=str(tmp_path / "bench_out"))
    assert out.endswith(".json")
    assert (tmp_path / "bench_out").exists()
