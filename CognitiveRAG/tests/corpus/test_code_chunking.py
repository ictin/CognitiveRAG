from CognitiveRAG.crag.corpus.chunkers import chunk_document


def test_code_chunking_is_symbol_aware():
    text = """
class Demo:
    def run(self):
        return "ok"

def helper(a, b):
    return a + b
"""
    chunks = chunk_document(
        document_id="doc_code",
        source_path="/tmp/demo.py",
        text=text,
        chunk_size=200,
        chunk_overlap=30,
        base_metadata={},
    )
    assert chunks
    assert all(c.metadata["source_format"] == "code" for c in chunks)
    assert all(c.metadata["chunk_strategy"] == "code_symbol" for c in chunks)
    names = {c.metadata.get("symbol_name") for c in chunks}
    assert "Demo" in names or "helper" in names
