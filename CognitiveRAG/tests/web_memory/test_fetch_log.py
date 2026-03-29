from CognitiveRAG.crag.web_memory.fetch_log import WebFetchLogStore


def test_fetch_log_records_attempts(tmp_path):
    store = WebFetchLogStore(tmp_path / "fetch_log.sqlite3")
    store.append(
        query="latest changelog",
        query_variant="latest changelog",
        status="ok",
        http_status=200,
        error=None,
        result_count=3,
        fetched_at="2026-03-29T13:00:00Z",
    )
    rows = store.list_recent(limit=5)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["result_count"] == 3
