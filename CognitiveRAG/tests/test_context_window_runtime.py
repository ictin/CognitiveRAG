def test_retriever_exposes_assemble_session_context(tmp_path):
    from CognitiveRAG.retriever import assemble_session_context
    # create raw messages fallback file used by assemble_context
    import os, json
    sess = 'rt_sess'
    raw = []
    for i in range(30):
        raw.append({'index': i, 'text': f'msg {i}', 'meta': {}})
    data_dir = os.path.join(os.getcwd(), 'data', 'session_memory')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, f'raw_{sess}.json'), 'w', encoding='utf-8') as f:
        json.dump(raw, f)

    ctx = assemble_session_context(sess, fresh_tail_count=5, budget=1000)
    assert 'fresh_tail' in ctx and 'summaries' in ctx
    assert len(ctx['fresh_tail']) == 5

    # cleanup
    os.remove(os.path.join(data_dir, f'raw_{sess}.json'))
