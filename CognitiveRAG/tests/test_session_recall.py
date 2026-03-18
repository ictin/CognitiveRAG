def test_session_search_describe_expand(tmp_path):
    from CognitiveRAG.session_memory import ConversationStore, MessagePartsStore, SummaryNodeStore, SummaryEdgeStore, LargeFileStore
    from CognitiveRAG.session_memory.recall import search_session_memory, describe_session_item, expand_session_item

    # prepare stores
    conv_db = tmp_path / "conversations.sqlite3"
    conv = ConversationStore(str(conv_db))
    conv.append_message('sess1', 'm1', 'user', 'The quick brown fox', '2026-03-18T01:00:00Z')
    conv.append_message('sess1', 'm2', 'user', 'jumps over lazy dog', '2026-03-18T01:01:00Z')

    parts_db = tmp_path / "parts.sqlite3"
    parts = MessagePartsStore(str(parts_db))
    parts.add_part('sess1', 'm1', 0, 'The quick', None)
    parts.add_part('sess1', 'm1', 1, 'brown fox', None)

    nodes_db = tmp_path / "nodes.sqlite3"
    nodes = SummaryNodeStore(str(nodes_db))
    nodes.upsert_node('n1', 'sess1', 'summary about fox', '{}', '2026-03-18T01:02:00Z')

    edges_db = tmp_path / "edges.sqlite3"
    edges = SummaryEdgeStore(str(edges_db))
    edges.add_edge('n1', 'n2', 'related', 1.0)

    lf_db = tmp_path / "files.sqlite3"
    lf = LargeFileStore(str(lf_db))
    lf.upsert_file('lf1', '/tmp/large.txt', '{}', '2026-03-18T01:03:00Z')

    # search
    results = search_session_memory('sess1', 'fox', db_prefix=str(tmp_path))
    assert isinstance(results, list)
    assert any(r['item_type'] in ('message','message_part','summary_node') for r in results)

    # pick a message ref and describe
    msg_refs = [r for r in results if r['item_type']=='message']
    if msg_refs:
        desc = describe_session_item(msg_refs[0], db_prefix=str(tmp_path))
        assert desc['item_type'] == 'message'
        ex = expand_session_item(msg_refs[0], db_prefix=str(tmp_path))
        # expansion should include message parts
        assert any(e['item_type']=='message_part' for e in ex)

    # summary node describe + expand
    sn_refs = [r for r in results if r['item_type']=='summary_node']
    if sn_refs:
        descn = describe_session_item(sn_refs[0], db_prefix=str(tmp_path))
        assert descn['item_type'] == 'summary_node'
        exn = expand_session_item(sn_refs[0], db_prefix=str(tmp_path))
        assert any(e['item_type']=='summary_edge' for e in exn)

    # large file describe and expand
    lref = {'item_type':'large_file', 'session_id':'sess1', 'primary_id':'lf1'}
    dlf = describe_session_item(lref, db_prefix=str(tmp_path))
    assert dlf['item_type'] == 'large_file'
    exlf = expand_session_item(lref, db_prefix=str(tmp_path))
    assert any(e['item_type']=='large_file_meta' for e in exlf)
