from CognitiveRAG.crag.promotion.lifecycle import normalize_reasoning_lifecycle, normalize_web_lifecycle


def test_web_lifecycle_covers_unreviewed_state():
    out = normalize_web_lifecycle(
        promotion_state="staged",
        freshness_lifecycle_state="stale",
        approved_at=None,
        last_validated_at=None,
        revalidation_requested_at=None,
    )
    assert out["lifecycle_state"] == "unreviewed"
    assert out["approval_state"] == "unreviewed"


def test_web_lifecycle_covers_revalidation_required_state():
    out = normalize_web_lifecycle(
        promotion_state="trusted",
        freshness_lifecycle_state="revalidation_pending",
        approved_at="2026-05-04T10:00:00Z",
        last_validated_at="2026-05-04T10:00:00Z",
        revalidation_requested_at="2026-05-05T10:00:00Z",
    )
    assert out["lifecycle_state"] == "revalidation_required"
    assert out["revalidation_state"] == "required"


def test_web_lifecycle_covers_revalidated_state():
    out = normalize_web_lifecycle(
        promotion_state="trusted",
        freshness_lifecycle_state="fresh",
        approved_at="2026-05-04T10:00:00Z",
        last_validated_at="2026-05-05T10:00:00Z",
        revalidation_requested_at=None,
    )
    assert out["lifecycle_state"] == "revalidated"
    assert out["approval_state"] == "approved"


def test_reasoning_lifecycle_marks_stale_as_revalidation_required():
    out = normalize_reasoning_lifecycle(
        freshness_state="stale",
        success_signal_count=0,
    )
    assert out["lifecycle_state"] == "stale"
    assert out["revalidation_state"] == "required"
