from __future__ import annotations

import re
from typing import Any

from CognitiveRAG.crag.contracts.schemas import ContextCandidate


_NEGATIVE_MARKERS = {"not", "never", "cannot", "cant", "can't", "failed", "failure", "no", "disabled", "off"}
_POSITIVE_MARKERS = {"works", "working", "succeeds", "success", "yes", "enabled", "active", "on"}


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", str(text or "").lower()) if len(tok) >= 3}


def _polarity(text: str) -> str:
    toks = _tokens(text)
    if {"not", "never", "cannot", "cant", "can't", "no"} & toks:
        return "negative"
    has_neg = bool(toks & _NEGATIVE_MARKERS)
    has_pos = bool(toks & _POSITIVE_MARKERS)
    if has_neg and not has_pos:
        return "negative"
    if has_pos and not has_neg:
        return "positive"
    return "mixed"


def compatibility_conflict_reason(candidate: ContextCandidate, selected: list[ContextCandidate]) -> str | None:
    """Heuristic compatibility gate (B3 partial, non-NLI).

    Priority order:
    1) explicit provenance claim key/value mismatch
    2) opposite polarity over overlapping lexical terms
    """
    cand_prov: dict[str, Any] = dict(candidate.provenance or {})
    cand_claim_key = str(cand_prov.get("claim_key") or "").strip().lower()
    cand_claim_value = str(cand_prov.get("claim_value") or "").strip().lower()

    cand_tokens = _tokens(candidate.text)
    cand_polarity = _polarity(candidate.text)

    for existing in selected:
        existing_prov: dict[str, Any] = dict(existing.provenance or {})
        ex_claim_key = str(existing_prov.get("claim_key") or "").strip().lower()
        ex_claim_value = str(existing_prov.get("claim_value") or "").strip().lower()

        if cand_claim_key and ex_claim_key and cand_claim_key == ex_claim_key:
            if cand_claim_value and ex_claim_value and cand_claim_value != ex_claim_value:
                return "compatibility_conflict"

        if cand_polarity == "mixed":
            continue
        ex_polarity = _polarity(existing.text)
        if ex_polarity == "mixed" or ex_polarity == cand_polarity:
            continue
        overlap = cand_tokens & _tokens(existing.text)
        if len(overlap) >= 2:
            return "compatibility_conflict"

    return None
