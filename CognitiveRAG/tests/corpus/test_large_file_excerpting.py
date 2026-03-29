from CognitiveRAG.crag.corpus.chunkers import chunk_document


def test_large_file_uses_excerpt_windows():
    text = ("abcdef " * 6000).strip()
    chunks = chunk_document(
        document_id="doc_large",
        source_path="/tmp/huge.txt",
        text=text,
        chunk_size=1000,
        chunk_overlap=100,
        base_metadata={},
    )
    assert len(chunks) > 1
    assert all(c.metadata["source_format"] == "large_file" for c in chunks)
    assert all(c.metadata["document_kind"] == "large_file_excerpt" for c in chunks)
    assert all(c.metadata.get("excerpt_window_total") is not None for c in chunks)
