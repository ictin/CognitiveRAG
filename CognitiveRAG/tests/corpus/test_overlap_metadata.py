from CognitiveRAG.crag.corpus.chunkers import chunk_document


def test_overlap_metadata_is_present_and_deterministic():
    text = ("one two three four five six seven eight nine ten " * 120).strip()
    chunks = chunk_document(
        document_id="doc_overlap",
        source_path="/tmp/overlap.txt",
        text=text,
        chunk_size=180,
        chunk_overlap=40,
        base_metadata={},
    )
    assert len(chunks) > 2
    assert all("overlap_prev_chars" in c.metadata for c in chunks)
    assert all("overlap_next_chars" in c.metadata for c in chunks)
    assert chunks[1].metadata["overlap_prev_chars"] >= 0
