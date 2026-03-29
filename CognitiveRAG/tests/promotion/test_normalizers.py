from CognitiveRAG.crag.promotion.normalizers import normalize_prescription, normalize_proposition


def test_normalize_proposition_has_stable_fields():
    unit = normalize_proposition("The user said that user prefers automation.", provenance={"session_id": "s1"})
    assert unit.kind == "proposition"
    assert unit.memory_subtype in {"profile_preference", "stable_fact"}
    assert unit.canonical_text
    assert unit.normalized_key.startswith("proposition:")
    assert unit.provenance.get("session_id") == "s1"


def test_normalize_prescription_has_stable_fields():
    unit = normalize_prescription("Successful workflow was ingest -> validate -> deploy.")
    assert unit.kind == "prescription"
    assert unit.memory_subtype in {"workflow_pattern", "procedure_pattern"}
    assert unit.normalized_key.startswith("prescription:")
    assert "ingest" in unit.canonical_text

