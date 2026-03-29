from pathlib import Path

from CognitiveRAG.core.settings import Settings
from CognitiveRAG.ingest.pipeline import IngestionPipeline


class _MemStore:
    def __init__(self):
        self.documents = []
        self.chunks = []

    def upsert_document(self, document_id: str, source_path: str, content_hash: str):
        self.documents.append((document_id, source_path, content_hash))

    def replace_chunks(self, document_id, chunks):
        self.chunks = list(chunks)


class _NoopVector:
    def __init__(self):
        self.payloads = []

    def upsert_chunks(self, payloads):
        self.payloads = list(payloads)


class _NoopLexical(_NoopVector):
    pass


class _NoopGraph:
    def upsert_extractions(self, document_id, extractions):
        return None


def test_ingest_pipeline_preserves_overlap_and_provenance_metadata(tmp_path):
    src = tmp_path / "demo.md"
    src.write_text("# Header\n\nA paragraph.\n\n## Details\nMore text here.", encoding="utf-8")

    settings = Settings()
    settings.retrieval.chunk_size = 80
    settings.retrieval.chunk_overlap = 16

    metadata_store = _MemStore()
    vector_store = _NoopVector()
    lexical_store = _NoopLexical()
    graph_store = _NoopGraph()

    pipeline = IngestionPipeline(
        settings=settings,
        metadata_store=metadata_store,
        vector_store=vector_store,
        lexical_store=lexical_store,
        graph_store=graph_store,
    )

    docs = pipeline.ingest_path(Path(src))
    assert len(docs) == 1
    assert metadata_store.chunks
    first = metadata_store.chunks[0]
    metadata_json = first[4]
    assert '"source_path"' in metadata_json
    assert '"chunk_char_start"' in metadata_json
    assert '"overlap_prev_chars"' in metadata_json
    assert '"provenance"' in metadata_json
