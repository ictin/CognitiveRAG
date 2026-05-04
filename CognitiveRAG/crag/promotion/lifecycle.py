from __future__ import annotations

from typing import Any, Dict


def normalize_web_lifecycle(
    *,
    promotion_state: str,
    freshness_lifecycle_state: str,
    approved_at: str | None,
    last_validated_at: str | None,
    revalidation_requested_at: str | None,
) -> Dict[str, Any]:
    state = str(promotion_state or "").strip().lower()
    freshness = str(freshness_lifecycle_state or "").strip().lower()
    approved = state == "trusted"
    validated = bool((last_validated_at or "").strip() or (approved_at or "").strip())
    revalidation_requested = bool((revalidation_requested_at or "").strip())

    if not approved:
        lifecycle_state = "unreviewed"
    elif freshness == "revalidation_pending":
        lifecycle_state = "revalidation_required"
    elif freshness == "stale":
        lifecycle_state = "stale"
    elif validated:
        lifecycle_state = "revalidated"
    else:
        lifecycle_state = "approved"

    return {
        "lifecycle_state": lifecycle_state,
        "approval_state": ("approved" if approved else "unreviewed"),
        "freshness_lifecycle_state": freshness or "stale",
        "revalidation_state": ("required" if freshness in {"stale", "revalidation_pending"} else "not_required"),
        "revalidation_requested": revalidation_requested,
        "validated": validated,
    }


def normalize_reasoning_lifecycle(
    *,
    freshness_state: str | None,
    success_signal_count: int,
) -> Dict[str, Any]:
    fresh = str(freshness_state or "").strip().lower()
    sig = int(success_signal_count or 0)
    if fresh in {"stale", "unknown"}:
        lifecycle_state = "stale"
    elif sig > 0:
        lifecycle_state = "revalidated"
    else:
        lifecycle_state = "approved"
    return {
        "lifecycle_state": lifecycle_state,
        "approval_state": "approved",
        "freshness_state": (fresh or "unknown"),
        "revalidation_state": ("required" if lifecycle_state == "stale" else "not_required"),
    }
