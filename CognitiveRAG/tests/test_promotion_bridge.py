import os
import json
from pathlib import Path
from CognitiveRAG.session_memory.promotion_bridge import promote_session_summaries
from CognitiveRAG.memory.reasoning_store import ReasoningStore


def test_promote_session_summaries_idempotent(tmp_path):
    session_id = 'prom_sess'
    data_dir = Path(os.getcwd()) / 'data' / 'session_memory'
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        {'chunk_index': 0, 'summary': 'This is first summary.'},
        {'chunk_index': 1, 'summary': 'Second summary text here.'},
    ]
    sum_file = data_dir / f'summaries_{session_id}.json'
    with open(sum_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f)

    db_path = tmp_path / 'reasoning.db'
    rs = ReasoningStore(db_path)

    # first promotion
    patterns = promote_session_summaries(session_id, reasoning_store=rs, dry_run=False)
    assert len(patterns) == 2

    # check store has entries
    out = rs.query('session:prom_sess')
    assert len(out) >= 1

    # second promotion (idempotent)
    patterns2 = promote_session_summaries(session_id, reasoning_store=rs, dry_run=False)
    assert len(patterns2) == 2

    # ensure no duplicates by querying latest and checking pattern_ids are stable
    ids_first = sorted([p.pattern_id for p in patterns])
    ids_second = sorted([p.pattern_id for p in patterns2])
    assert ids_first == ids_second

    # raw summaries file unchanged
    with open(sum_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert loaded == summaries

    # cleanup
    sum_file.unlink()
