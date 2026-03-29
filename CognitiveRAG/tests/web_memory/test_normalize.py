from CognitiveRAG.crag.web_memory.normalize import normalize_web_result


def test_normalize_preserves_canonical_fields():
    raw = {
        "title": "Example",
        "href": "https://example.com/a",
        "body": "Snippet text",
        "date": "2026-03-20",
    }
    out = normalize_web_result(
        raw=raw,
        query="latest example",
        query_variant="latest example",
        rank=0,
        fetched_at="2026-03-29T12:00:00Z",
    )
    assert out["title"] == "Example"
    assert out["url"] == "https://example.com/a"
    assert out["snippet"] == "Snippet text"
    assert out["fetched_at"] == "2026-03-29T12:00:00Z"
    assert out["content_hash"]
    assert out["source_id"]
