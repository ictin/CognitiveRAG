import sqlite3
from pathlib import Path

from CognitiveRAG.memory.reasoning_store import ReasoningStore
from CognitiveRAG.schemas.memory import ReasoningPattern
from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve as retrieve_promoted


def _pattern(
    pattern_id: str,
    signature: str,
    summary: str,
    *,
    confidence: float = 0.8,
    provenance: list[str] | None = None,
    normalized_text: str | None = None,
) -> ReasoningPattern:
    return ReasoningPattern(
        pattern_id=pattern_id,
        problem_signature=signature,
        reasoning_steps=["step-a", "step-b"],
        solution_summary=summary,
        confidence=confidence,
        provenance=provenance or [],
        normalized_text=normalized_text or summary.lower(),
    )


def _row_for_pattern(db_path: Path, pattern_id: str):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT pattern_id, reuse_count, provenance_json, merged_from_json, exact_fingerprint, near_duplicate_of, canonical_pattern_id "
            "FROM reasoning_patterns WHERE pattern_id=?",
            (pattern_id,),
        ).fetchone()


def test_reasoning_store_exact_duplicate_increments_reuse_and_merges_provenance(tmp_path: Path):
    db_path = tmp_path / "reasoning.sqlite3"
    store = ReasoningStore(db_path)
    store.upsert(
        _pattern(
            "rp_exact_a",
            "debug timeout path",
            "apply bounded diagnosis and verify fallback",
            provenance=["src://a"],
        )
    )
    store.upsert(
        _pattern(
            "rp_exact_b",
            "debug timeout path",
            "apply bounded diagnosis and verify fallback",
            provenance=["src://b"],
        )
    )

    row_a = _row_for_pattern(db_path, "rp_exact_a")
    row_b = _row_for_pattern(db_path, "rp_exact_b")
    assert row_a is not None
    assert row_b is None, "exact duplicate should fold into canonical record instead of a separate row"
    assert int(row_a["reuse_count"]) == 2
    assert "src://a" in (row_a["provenance_json"] or "")
    assert "src://b" in (row_a["provenance_json"] or "")
    assert "rp_exact_a" in (row_a["merged_from_json"] or "")
    assert "rp_exact_b" in (row_a["merged_from_json"] or "")


def test_reasoning_store_near_duplicate_links_without_merging(tmp_path: Path):
    db_path = tmp_path / "reasoning.sqlite3"
    store = ReasoningStore(db_path)
    store.upsert(
        _pattern(
            "rp_near_a",
            "stale plugin runtime path mismatch",
            "apply bounded diagnosis and verify fallback.",
            provenance=["src://a"],
            normalized_text="apply bounded diagnosis and verify fallback.",
        )
    )
    store.upsert(
        _pattern(
            "rp_near_b",
            "stale plugin runtime path mismatch",
            "apply bounded diagnosis and verify fallback",
            provenance=["src://b"],
            normalized_text="apply bounded diagnosis and verify fallback",
        )
    )
    row_a = _row_for_pattern(db_path, "rp_near_a")
    row_b = _row_for_pattern(db_path, "rp_near_b")
    assert row_a is not None and row_b is not None
    assert row_b["near_duplicate_of"] == "rp_near_a"
    assert row_b["canonical_pattern_id"] == "rp_near_a"
    assert int(row_a["reuse_count"]) == 1
    assert int(row_b["reuse_count"]) == 1

    store.upsert(
        _pattern(
            "rp_near_c",
            "stale plugin runtime path mismatch",
            "apply bounded diagnosis and verify fallback!",
            provenance=["src://c"],
            normalized_text="apply bounded diagnosis and verify fallback!",
        )
    )
    row_c = _row_for_pattern(db_path, "rp_near_c")
    assert row_c is not None
    assert row_c["near_duplicate_of"] == "rp_near_a", "near-duplicate linking should be deterministic"


def test_reasoning_store_distinct_patterns_are_not_overmerged(tmp_path: Path):
    db_path = tmp_path / "reasoning.sqlite3"
    store = ReasoningStore(db_path)
    store.upsert(
        _pattern(
            "rp_distinct_a",
            "reduce timeout regressions",
            "use bounded discovery and stale-session cleanup",
            provenance=["src://a"],
        )
    )
    store.upsert(
        _pattern(
            "rp_distinct_b",
            "improve hook retention in short videos",
            "front-load concrete payoff and keep visual beats fast",
            provenance=["src://b"],
        )
    )
    row_a = _row_for_pattern(db_path, "rp_distinct_a")
    row_b = _row_for_pattern(db_path, "rp_distinct_b")
    assert row_a is not None and row_b is not None
    assert row_a["near_duplicate_of"] is None
    assert row_b["near_duplicate_of"] is None
    assert row_a["exact_fingerprint"] != row_b["exact_fingerprint"]


def test_reasoning_store_repeated_same_pattern_id_increments_reuse_count(tmp_path: Path):
    db_path = tmp_path / "reasoning.sqlite3"
    store = ReasoningStore(db_path)
    pattern = _pattern(
        "rp_repeat",
        "repeat-safe canonicalization",
        "keep one canonical pattern and increment reuse count",
        provenance=["src://1"],
    )
    store.upsert(pattern)
    store.upsert(pattern)
    store.upsert(pattern)

    row = _row_for_pattern(db_path, "rp_repeat")
    assert row is not None
    assert int(row["reuse_count"]) == 3


def test_promoted_lane_reuse_count_boost_is_additive_and_bounded(tmp_path: Path):
    db_path = tmp_path / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT, "
            "exact_fingerprint TEXT, near_fingerprint TEXT, canonical_pattern_id TEXT, near_duplicate_of TEXT, reuse_count INTEGER, merged_from_json TEXT)"
        )
        for pid, reuse_count in (("rp_low", 1), ("rp_high", 5)):
            db.execute(
                "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pid,
                    "debug timeout path",
                    "[]",
                    "apply bounded diagnosis and verify fallback",
                    0.7,
                    "[]",
                    "workflow_pattern",
                    "apply bounded diagnosis and verify fallback",
                    "current",
                    None,
                    None,
                    pid,
                    None,
                    reuse_count,
                    "[]",
                ),
            )
        db.commit()

    hits = retrieve_promoted(
        workdir=str(tmp_path),
        intent_family=IntentFamily.INVESTIGATIVE,
        query="debug timeout fallback",
        top_k=2,
    )
    assert [h.id for h in hits] == ["promoted:rp_high", "promoted:rp_low"]
    assert hits[0].provenance.get("reuse_count") == 5
    assert hits[1].provenance.get("reuse_count") == 1
    semantic_delta = hits[0].semantic_score - hits[1].semantic_score
    assert semantic_delta <= 0.18
    assert semantic_delta > 0.0


def test_promoted_lane_fallback_without_reuse_count_column(tmp_path: Path):
    db_path = tmp_path / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT)"
        )
        db.execute(
            "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?)",
            ("rp_legacy", "legacy", "[]", "legacy record", 0.6, "[]"),
        )
        db.commit()

    hits = retrieve_promoted(
        workdir=str(tmp_path),
        intent_family=IntentFamily.MEMORY_SUMMARY,
        query="legacy",
        top_k=1,
    )
    assert hits
    assert hits[0].provenance.get("reuse_count") == 1
