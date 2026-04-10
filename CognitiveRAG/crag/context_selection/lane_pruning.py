from __future__ import annotations

from typing import Iterable, List

from CognitiveRAG.crag.contracts.enums import RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _split_candidate(candidate: ContextCandidate, max_tokens: int) -> List[ContextCandidate]:
    if candidate.tokens <= max_tokens:
        return [candidate]
    words = (candidate.text or "").split()
    if not words:
        return [candidate]
    out: List[ContextCandidate] = []
    chunk_words: list[str] = []
    chunk_index = 0
    for word in words:
        chunk_words.append(word)
        text = " ".join(chunk_words)
        est = max(1, (len(text) + 3) // 4)
        if est >= max_tokens:
            part = candidate.model_copy(deep=True)
            part.id = f"{candidate.id}#part{chunk_index}"
            part.text = text
            part.tokens = est
            out.append(part)
            chunk_words = []
            chunk_index += 1
    if chunk_words:
        text = " ".join(chunk_words)
        part = candidate.model_copy(deep=True)
        part.id = f"{candidate.id}#part{chunk_index}"
        part.text = text
        part.tokens = max(1, (len(text) + 3) // 4)
        out.append(part)
    return out


def prune_lane_local(candidates: Iterable[ContextCandidate], max_candidate_tokens: int = 320) -> List[ContextCandidate]:
    """Deterministic lane-local pruning:
    1) near-duplicate pruning by normalized text per lane,
    2) deterministic split for oversized compressible candidates,
    3) tiny adjacent merge by lane/cluster where possible.
    """
    seen_by_lane: dict[str, set[str]] = {}
    deduped: List[ContextCandidate] = []

    for candidate in candidates:
        lane = candidate.lane.value
        norm = _norm(candidate.text)
        lane_seen = seen_by_lane.setdefault(lane, set())
        if norm in lane_seen:
            continue
        lane_seen.add(norm)
        if candidate.compressible and candidate.tokens > max_candidate_tokens:
            deduped.extend(_split_candidate(candidate, max_candidate_tokens))
        else:
            deduped.append(candidate)

    merged: List[ContextCandidate] = []
    for candidate in deduped:
        if (
            merged
            and candidate.compressible
            and merged[-1].compressible
            and merged[-1].lane == candidate.lane
            and candidate.lane != RetrievalLane.FRESH_TAIL
            and merged[-1].cluster_id == candidate.cluster_id
            and merged[-1].tokens < 32
            and candidate.tokens < 32
        ):
            combined = merged[-1].model_copy(deep=True)
            combined.id = f"{merged[-1].id}+{candidate.id}"
            combined.text = f"{merged[-1].text}\n{candidate.text}".strip()
            combined.tokens = max(1, (len(combined.text) + 3) // 4)
            merged[-1] = combined
        else:
            merged.append(candidate)

    return merged
