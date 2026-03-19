import os, json
from CognitiveRAG.session_memory.context_window import compact_session, assemble_context


def test_compact_and_assemble(tmp_path):
    session_id = 'sess-integ'
    workdir = os.path.join(os.getcwd(), 'data', 'session_memory')
    os.makedirs(workdir, exist_ok=True)
    raw_path = os.path.join(workdir, f'raw_{session_id}.json')

    # create 5 messages with indexes 0..4
    msgs = []
    for i in range(5):
        msgs.append({'message_id': f'm{i}', 'index': i, 'sender': 'user' if i%2==0 else 'assistant', 'text': f'text-{i}'})
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(msgs, f)

    # compact messages older than index 3 (i.e., messages 0,1,2)
    created = compact_session(session_id, older_than_index=3)
    assert isinstance(created, list)
    assert len(created) >= 1

    # ensure fallback summaries file exists
    sums_path = os.path.join(workdir, f'summaries_{session_id}.json')
    assert os.path.exists(sums_path)
    with open(sums_path, 'r', encoding='utf-8') as f:
        sums = json.load(f)
    assert len(sums) >= 1

    # assemble context
    ctx = assemble_context(session_id, fresh_tail_count=2, budget=1000)
    assert 'fresh_tail' in ctx and 'summaries' in ctx
    # fresh_tail should include the newest two messages (indexes 3 and 4)
    tail_indexes = [int(m.get('index')) for m in ctx['fresh_tail']]
    assert sorted(tail_indexes) == [3,4]
    # summaries should include at least one summary covering older messages
    assert len(ctx['summaries']) >= 1
