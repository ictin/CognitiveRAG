from __future__ import annotations

import re
from typing import Iterable, List

from CognitiveRAG.crag.contracts.schemas import ContradictionRecord, DiscoveryEvidenceRef


_NEGATIVE_MARKERS = {'not', 'never', 'cannot', 'cant', "can't", 'failed', 'failure', 'no', 'disabled', 'off'}
_POSITIVE_MARKERS = {'works', 'working', 'succeeds', 'success', 'yes', 'enabled', 'active', 'on'}


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.split(r'[^a-z0-9]+', str(text or '').lower()) if len(tok) >= 3}


def _polarity(text: str) -> str:
    toks = _tokens(text)
    if {'not', 'never', 'cannot', 'cant', "can't", 'no'} & toks:
        return 'negative'
    has_neg = bool(toks & _NEGATIVE_MARKERS)
    has_pos = bool(toks & _POSITIVE_MARKERS)
    if has_neg and not has_pos:
        return 'negative'
    if has_pos and not has_neg:
        return 'positive'
    return 'mixed'


def detect_contradictions(evidence: Iterable[DiscoveryEvidenceRef], min_overlap: int = 2) -> List[ContradictionRecord]:
    entries = list(evidence)
    out: List[ContradictionRecord] = []
    for i, left in enumerate(entries):
        left_tokens = _tokens(left.text)
        left_polarity = _polarity(left.text)
        if left_polarity == 'mixed':
            continue
        for right in entries[i + 1 :]:
            right_tokens = _tokens(right.text)
            overlap = left_tokens & right_tokens
            if len(overlap) < min_overlap:
                continue
            right_polarity = _polarity(right.text)
            if right_polarity == 'mixed' or right_polarity == left_polarity:
                continue
            strength = min(1.0, 0.3 + 0.1 * len(overlap))
            out.append(
                ContradictionRecord(
                    left_evidence_id=left.evidence_id,
                    right_evidence_id=right.evidence_id,
                    reason=f'opposite polarity over overlapping terms: {", ".join(sorted(list(overlap))[:5])}',
                    strength=round(strength, 4),
                )
            )
    return out
