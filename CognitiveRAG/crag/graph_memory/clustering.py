from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from CognitiveRAG.crag.retrieval.models import LaneHit


_WORD_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class ClusterAssignment:
    cluster_id: str
    cluster_label: str
    terms: Tuple[str, ...]


def _terms(text: str, *, limit: int = 4) -> Tuple[str, ...]:
    raw = [w for w in _WORD_RE.findall((text or "").lower()) if w and w not in _STOPWORDS]
    if not raw:
        return ("uncategorized",)
    uniq = sorted(dict.fromkeys(raw))
    return tuple(uniq[: max(1, int(limit))])


def _cluster_seed(hit: LaneHit) -> Tuple[str, ...]:
    prov = dict(hit.provenance or {})
    topic_rows = list(((prov.get("topic_graph") or {}).get("topics")) or [])
    cat_rows = list(((prov.get("category_graph") or {}).get("categories")) or [])
    topic_names = tuple(sorted(str(r.get("topic") or "") for r in topic_rows if str(r.get("topic") or "")))
    category_names = tuple(sorted(str(r.get("category") or "") for r in cat_rows if str(r.get("category") or "")))
    return (
        hit.lane.value,
        hit.memory_type.value,
        *topic_names[:2],
        *category_names[:2],
        *_terms(hit.text),
    )


def assignment_for_hit(hit: LaneHit) -> ClusterAssignment:
    seed = tuple(s for s in _cluster_seed(hit) if s)
    normalized = "|".join(seed) if seed else "uncategorized"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    cluster_id = f"cl:{digest}"
    terms = seed[2:] if len(seed) > 2 else _terms(hit.text)
    label = " / ".join(terms[:2]) if terms else "uncategorized"
    return ClusterAssignment(cluster_id=cluster_id, cluster_label=label, terms=tuple(terms[:4]))


def annotate_hits_with_clusters(hits: Iterable[LaneHit]) -> List[LaneHit]:
    out: List[LaneHit] = []
    for hit in hits:
        clone = hit.model_copy(deep=True)
        assignment = assignment_for_hit(clone)
        clone.cluster_id = assignment.cluster_id
        prov = dict(clone.provenance or {})
        prov["clustering_helper"] = {
            "cluster_id": assignment.cluster_id,
            "cluster_label": assignment.cluster_label,
            "cluster_terms": list(assignment.terms),
            "helper_only": True,
            "authoritative": False,
        }
        clone.provenance = prov
        out.append(clone)
    return out
