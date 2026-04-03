from CognitiveRAG.crag.contracts.enums import MemoryType
from CognitiveRAG.crag.retrieval.semantic_lane import retrieve
from CognitiveRAG.crag.retrieval.vector_backend import DEFAULT_VECTOR_BACKEND, TokenJaccardVectorBackend, VectorRecord


def test_vector_backend_adapter_scores_deterministically():
    backend = TokenJaccardVectorBackend()
    records = [
        VectorRecord(record_id="r1", text="cache key normalization strategy", memory_type=MemoryType.EPISODIC_RAW, source_type="fresh_tail"),
        VectorRecord(record_id="r2", text="unrelated corpus note", memory_type=MemoryType.EPISODIC_RAW, source_type="fresh_tail"),
    ]
    matches = backend.search(query="cache key normalization", records=records, top_k=4)
    assert [m.record.record_id for m in matches] == ["r1"]
    assert matches[0].score > 0


def test_semantic_lane_contract_shape():
    hits = retrieve(
        query="memory organization",
        session_id="s2",
        fresh_tail=[{"text": "memory organized in layers"}],
        older_raw=[],
        summaries=[{"summary": "layered memory overview"}],
        top_k=4,
    )
    assert hits
    first = hits[0]
    assert first.lane.value == "semantic"
    assert first.semantic_score >= 0
    assert first.tokens > 0
    vb = dict((first.provenance or {}).get("vector_backend") or {})
    assert vb.get("abstraction_used") is True
    assert vb.get("selected_backend") == DEFAULT_VECTOR_BACKEND


def test_semantic_lane_unknown_backend_falls_back_safely():
    hits = retrieve(
        query="deterministic cache keys",
        session_id="s3",
        fresh_tail=[{"text": "deterministic cache keying approach"}],
        older_raw=[],
        summaries=[],
        top_k=4,
        vector_backend_name="unknown_backend_v0",
    )
    assert hits
    vb = dict((hits[0].provenance or {}).get("vector_backend") or {})
    assert vb.get("abstraction_used") is True
    assert vb.get("fallback_used") is True
    assert vb.get("selected_backend") == DEFAULT_VECTOR_BACKEND
