from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Protocol

from CognitiveRAG.crag.contracts.enums import MemoryType


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    text: str
    memory_type: MemoryType
    cluster_id: str | None = None
    source_type: str = "unknown"
    provenance: dict | None = None


@dataclass(frozen=True)
class VectorMatch:
    record: VectorRecord
    score: float
    backend: str
    debug: dict


class VectorBackend(Protocol):
    name: str

    def search(
        self,
        *,
        query: str,
        records: Iterable[VectorRecord],
        top_k: int,
        where: Mapping[str, str] | None = None,
    ) -> list[VectorMatch]:
        ...


def _norm_words(text: str) -> set[str]:
    return {w for w in " ".join((text or "").lower().split()).split() if w}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(max(1, len(a | b)))


class TokenJaccardVectorBackend:
    """Deterministic in-process vector backend adapter.

    This is the current default implementation. It is intentionally simple,
    backend-owned, and interface-compatible with future backend adapters.
    """

    name = "token_jaccard_v1"

    def search(
        self,
        *,
        query: str,
        records: Iterable[VectorRecord],
        top_k: int,
        where: Mapping[str, str] | None = None,
    ) -> list[VectorMatch]:
        qwords = _norm_words(query)
        out: list[VectorMatch] = []
        where_map: Mapping[str, str] = dict(where or {})

        for record in list(records):
            if where_map:
                source_filter = str(where_map.get("source_type") or "")
                if source_filter and source_filter != str(record.source_type):
                    continue
            score = _jaccard(qwords, _norm_words(record.text))
            if score <= 0:
                continue
            out.append(
                VectorMatch(
                    record=record,
                    score=float(score),
                    backend=self.name,
                    debug={
                        "source_type": record.source_type,
                        "cluster_id": record.cluster_id,
                    },
                )
            )

        out.sort(key=lambda row: (-float(row.score), row.record.record_id))
        return out[: max(1, int(top_k))]


_VECTOR_BACKENDS: Dict[str, VectorBackend] = {
    TokenJaccardVectorBackend.name: TokenJaccardVectorBackend(),
}

DEFAULT_VECTOR_BACKEND = TokenJaccardVectorBackend.name


def resolve_vector_backend(name: str | None) -> tuple[VectorBackend, str, bool]:
    requested = str(name or DEFAULT_VECTOR_BACKEND)
    backend = _VECTOR_BACKENDS.get(requested)
    if backend is not None:
        return backend, requested, False
    fallback = _VECTOR_BACKENDS[DEFAULT_VECTOR_BACKEND]
    return fallback, requested, True

