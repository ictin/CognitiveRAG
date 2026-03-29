from CognitiveRAG.crag.corpus.chunkers import chunk_document


def test_markdown_chunking_is_section_aware():
    text = """# Intro
This is intro text.

## Details
- item 1
- item 2

## Notes
Final section with extra details for chunking."""
    chunks = chunk_document(
        document_id="doc_md",
        source_path="/tmp/doc.md",
        text=text,
        chunk_size=120,
        chunk_overlap=20,
        base_metadata={},
    )
    assert chunks
    assert all(c.metadata["source_format"] == "markdown" for c in chunks)
    assert all(c.metadata["chunk_strategy"] == "markdown_section" for c in chunks)
    titles = {c.metadata.get("section_title") for c in chunks}
    assert "Intro" in titles or "preamble" in titles
    assert "Details" in titles
