from __future__ import annotations

from typing import Any, Dict


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def normalized_lifecycle_view(*, source_class: str, provenance: Dict[str, Any] | None = None) -> Dict[str, Any]:
    prov = dict(provenance or {})
    state = _norm(prov.get("promotion_state"))
    freshness = _norm(prov.get("freshness_lifecycle_state") or prov.get("freshness_state"))
    trust_status = _norm(prov.get("trust_status"))
    approval = _norm(prov.get("approval_status"))
    import_state = _norm(prov.get("import_state"))
    authoritative = bool(prov.get("authoritative"))

    normalized_state = "unreviewed"
    if import_state == "quarantined" or trust_status == "untrusted":
        normalized_state = "quarantined"
    elif approval in {"rejected", "retired"}:
        normalized_state = approval
    elif freshness in {"revalidation_pending", "revalidation_required"}:
        normalized_state = "revalidation_required"
    elif freshness == "stale":
        normalized_state = "stale"
    elif state == "trusted" or trust_status == "trusted" or approval in {"approved", "trusted"}:
        normalized_state = "trusted"
        if freshness == "revalidated":
            normalized_state = "revalidated"
    elif state == "staged":
        normalized_state = "unreviewed"

    return {
        "source_class": str(source_class or ""),
        "normalized_state": normalized_state,
        "promotion_state": state or None,
        "freshness_lifecycle_state": freshness or None,
        "trust_status": trust_status or None,
        "approval_status": approval or None,
        "import_state": import_state or None,
        "authoritative": authoritative,
        "helper_only": bool(prov.get("helper_only")),
    }
