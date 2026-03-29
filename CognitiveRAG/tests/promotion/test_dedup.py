from CognitiveRAG.crag.promotion.dedup import dedup_units
from CognitiveRAG.crag.promotion.normalizers import normalize_proposition


def test_dedup_equivalent_units_collapse():
    a = normalize_proposition("user prefers concise answers", confidence=0.6, provenance={"source": "a"})
    b = normalize_proposition("The user said that user prefers concise answers.", confidence=0.9, provenance={"source": "b"})
    out = dedup_units([a, b])
    assert len(out) == 1
    assert out[0].confidence >= 0.9
    assert int(out[0].provenance.get("dedup_count", 1)) >= 2

