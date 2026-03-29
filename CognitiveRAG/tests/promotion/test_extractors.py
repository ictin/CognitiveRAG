from CognitiveRAG.crag.promotion.extractors import extract_prescriptions, extract_propositions


def test_extractors_capture_proposition_and_prescription():
    text = (
        "The user said that the user prefers concise technical answers. "
        "We concluded that the successful workflow was ingest -> validate -> deploy."
    )
    props = extract_propositions(text)
    presc = extract_prescriptions(text)
    assert any("prefer" in p.lower() for p in props)
    assert any(("workflow" in p.lower()) or ("->" in p) for p in presc)

