import os, json
from pathlib import Path
from CognitiveRAG.session_memory.promotion_bridge import promote_session_summaries
from CognitiveRAG.memory.reasoning_store import ReasoningStore
from CognitiveRAG.retriever import HybridRetriever
from CognitiveRAG.retrieval_modes import RetrievalMode


def test_promoted_summaries_are_reused(tmp_path):
    session_id = 'reuse_sess'
    data_dir = Path(os.getcwd()) / 'data' / 'session_memory'
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        {'chunk_index': 0, 'summary': 'Reusable summary content about X.'},
    ]
    sum_file = data_dir / f'summaries_{session_id}.json'
    with open(sum_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f)

    # create reasoning store
    db_path = tmp_path / 'reasoning.db'
    rs = ReasoningStore(db_path)

    # promote via helper (real bridge)
    patterns = promote_session_summaries(session_id, reasoning_store=rs, dry_run=False)
    assert len(patterns) == 1

    # create a retriever with injected reasoning_store
    class DummyKB:
        bm25_index = None
        doc_store = {}
        vector_store = None
    retriever = HybridRetriever(DummyKB(), reasoning_store=rs)

    # run retrieval in TASK_MEMORY mode which enables task_profile_reasoning
    docs = retriever.retrieve('summary content about X', top_k=5, mode=RetrievalMode.TASK_MEMORY)
    # docs should include reasoning-derived documents
    found = False
    for d in docs:
        meta = getattr(d, 'metadata', {}) or {}
        if meta.get('source_type') == 'reasoning' or meta.get('source_type') == 'reasoning':
            found = True
            break
    assert found, "Promoted reasoning item not found in retrieval results"

    # idempotent promotion
    patterns2 = promote_session_summaries(session_id, reasoning_store=rs, dry_run=False)
    assert [p.pattern_id for p in patterns] == [p.pattern_id for p in patterns2]

    # cleanup
    sum_file.unlink()
