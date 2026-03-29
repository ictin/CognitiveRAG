from CognitiveRAG.crag.corpus.chunkers import chunk_document


def test_chunk_provenance_has_path_and_location_anchors():
    text = "alpha\nbeta\ngamma\ndelta\nepsilon"
    chunks = chunk_document(
        document_id="doc_prov",
        source_path="/tmp/prov.md",
        text=text,
        chunk_size=20,
        chunk_overlap=5,
        base_metadata={},
    )
    assert chunks
    for c in chunks:
        prov = c.metadata.get("provenance") or {}
        assert prov.get("source_path") == "/tmp/prov.md"
        assert isinstance(prov.get("char_start"), int)
        assert isinstance(prov.get("char_end"), int)
        assert isinstance(prov.get("line_start"), int)
        assert isinstance(prov.get("line_end"), int)
