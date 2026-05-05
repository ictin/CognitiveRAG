from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from CognitiveRAG.crag.contracts.enums import RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, RoleProbe


@dataclass(frozen=True)
class HelperBranchSuggestion:
    helper_source_type: str
    branch_reason: str
    branch_prompt: str
    strength: float
    provenance_refs: List[Dict[str, str]]


def _bucket_from_candidate(candidate: ContextCandidate) -> List[HelperBranchSuggestion]:
    prov = dict(candidate.provenance or {})
    out: List[HelperBranchSuggestion] = []
    cid = str(candidate.id or "")
    lane = candidate.lane.value

    category_graph = dict(prov.get("category_graph") or {})
    categories = list(category_graph.get("categories") or [])
    if categories:
        top = categories[0]
        cname = str(top.get("category") or "unknown_category")
        score = float(top.get("score") or 0.0)
        out.append(
            HelperBranchSuggestion(
                helper_source_type="category_graph",
                branch_reason=f"category:{cname}",
                branch_prompt=f"Explore non-obvious risks and contradictions around category '{cname}'.",
                strength=max(0.1, min(1.0, score)),
                provenance_refs=[{"type": "candidate", "id": cid, "lane": lane}],
            )
        )

    topic_graph = dict(prov.get("topic_graph") or {})
    topics = list(topic_graph.get("topics") or [])
    if topics:
        top = topics[0]
        tname = str(top.get("topic") or "unknown_topic")
        score = float(top.get("score") or 0.0)
        out.append(
            HelperBranchSuggestion(
                helper_source_type="topic_graph",
                branch_reason=f"topic:{tname}",
                branch_prompt=f"Check weak signals and contradiction edges around topic '{tname}'.",
                strength=max(0.1, min(1.0, score)),
                provenance_refs=[{"type": "candidate", "id": cid, "lane": lane}],
            )
        )

    clustering = dict(prov.get("clustering_helper") or {})
    cluster_id = str(clustering.get("cluster_id") or "")
    if cluster_id:
        out.append(
            HelperBranchSuggestion(
                helper_source_type="clustering",
                branch_reason=f"cluster:{cluster_id}",
                branch_prompt="Probe alternative interpretations within this cluster and surface conflicts.",
                strength=0.45,
                provenance_refs=[{"type": "candidate", "id": cid, "lane": lane}],
            )
        )

    return out


def suggest_graph_assisted_branches(
    *,
    candidate_pool: Iterable[ContextCandidate],
    max_suggestions: int = 3,
) -> List[HelperBranchSuggestion]:
    rows: List[HelperBranchSuggestion] = []
    for cand in candidate_pool:
        rows.extend(_bucket_from_candidate(cand))
    rows.sort(key=lambda r: (-float(r.strength), r.helper_source_type, r.branch_reason))

    dedup: List[HelperBranchSuggestion] = []
    seen = set()
    for row in rows:
        key = (row.helper_source_type, row.branch_reason)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
        if len(dedup) >= max(1, int(max_suggestions)):
            break
    return dedup


def suggestions_to_probes(suggestions: List[HelperBranchSuggestion]) -> List[RoleProbe]:
    probes: List[RoleProbe] = []
    for idx, suggestion in enumerate(suggestions):
        lanes = [RetrievalLane.SEMANTIC, RetrievalLane.CORPUS, RetrievalLane.EPISODIC]
        probes.append(
            RoleProbe(
                role=f"graph-helper-{suggestion.helper_source_type}",
                prompt=suggestion.branch_prompt,
                purpose=f"graph_helper:{suggestion.helper_source_type}:{suggestion.branch_reason}",
                expected_lanes=lanes,
                priority=max(1, 3 - idx),
            )
        )
    return probes
