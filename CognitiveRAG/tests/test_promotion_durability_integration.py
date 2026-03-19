import os, json
from pathlib import Path
from CognitiveRAG.session_memory.context_window import compact_session
from CognitiveRAG.session_memory.promotion_bridge import promote_session_summaries
from CognitiveRAG.memory.reasoning_store import ReasoningStore


def test_compact_then_promote(tmp_path):
    session_id = 'prom_integ'
    workdir = Path(os.getcwd()) / 'data' / 'session_memory'
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / f'raw_{session_id}.json'

    # write 4 messages indexes 0..3
    msgs = []
    for i in range(4):
        msgs.append({'message_id': f'm{i}', 'index': i, 'sender': 'user' if i%2==0 else 'assistant', 'text': f'content-{i}'})
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(msgs, f)

    # compact older than index 2 -> messages 0 and 1 summarized
    created = compact_session(session_id, older_than_index=2)
    assert len(created) >= 1

    # ensure summaries are present in fallback file
    sum_file = workdir / f'summaries_{session_id}.json'
    assert sum_file.exists()

    # promote into reasoning store
    db_path = tmp_path / 'reasoning.db'
    rs = ReasoningStore(db_path)
    patterns = promote_session_summaries(session_id, reasoning_store=rs, dry_run=False)
    assert len(patterns) >= 1

    # verify reasoning store query can find the promoted summary
    out = rs.query('session:prom_integ')
    assert len(out) >= 1
    # verify latest_chunk returns a dict with text matching summary
    lc = rs.latest_chunk()
    assert isinstance(lc, dict)
    assert 'text' in lc and lc['text']

    # cleanup
    raw_path.unlink()
    sum_file.unlink()
