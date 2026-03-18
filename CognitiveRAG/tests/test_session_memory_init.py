def test_session_memory_imports_and_init(tmp_path):
    from CognitiveRAG.session_memory import ConversationStore, MessagePartsStore, SummaryNodeStore, SummaryEdgeStore, ContextItemStore, LargeFileStore

    # instantiate stores with tmp_path DB paths
    conv_db = tmp_path / "conversations.sqlite3"
    conv = ConversationStore(str(conv_db))
    conv.append_message('s1', 'm1', 'user', 'hello', '2026-03-18T00:00:00Z')
    msgs = conv.get_messages('s1')
    assert isinstance(msgs, list) and msgs[0]['text'] == 'hello'

    parts_db = tmp_path / "parts.sqlite3"
    parts = MessagePartsStore(str(parts_db))
    parts.add_part('s1', 'm1', 0, 'hello part', None)
    ps = parts.get_parts('s1', 'm1')
    assert isinstance(ps, list) and ps[0]['text'] == 'hello part'

    nodes_db = tmp_path / "nodes.sqlite3"
    nodes = SummaryNodeStore(str(nodes_db))
    nodes.upsert_node('n1', 's1', 'summary', '{}', '2026-03-18T00:00:01Z')
    n = nodes.get_node('n1')
    assert n['text'] == 'summary'

    edges_db = tmp_path / "edges.sqlite3"
    edges = SummaryEdgeStore(str(edges_db))
    edges.add_edge('n1', 'n2', 'rel', 1.0)
    es = edges.get_edges_from('n1')
    assert isinstance(es, list) and es[0]['to_id'] == 'n2'

    context_db = tmp_path / "context.sqlite3"
    ctx = ContextItemStore(str(context_db))
    ctx.upsert_item('i1', 's1', 'type', '{}', '2026-03-18T00:00:02Z')
    it = ctx.get_item('i1')
    assert it['type'] == 'type'

    lf_db = tmp_path / "files.sqlite3"
    lf = LargeFileStore(str(lf_db))
    lf.upsert_file('f1', '/tmp/file', '{}', '2026-03-18T00:00:03Z')
    f = lf.get_file('f1')
    assert f['file_path'] == '/tmp/file'
