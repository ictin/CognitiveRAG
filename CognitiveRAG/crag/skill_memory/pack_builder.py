from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from CognitiveRAG.crag.graph_memory.skill_graph import read_skill_graph_signal
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.skill_memory.ranking import RankedArtifact
from CognitiveRAG.crag.skill_memory.retrieval import retrieve_skill_artifacts
from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact, SkillPack, SkillPackRequest
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


TYPE_ORDER = ["principle", "template", "example", "rubric", "anti_pattern", "workflow", "raw_chunk"]

AGENT_QUOTAS: Dict[str, Dict[str, int]] = {
    "script_agent": {
        "principle": 5,
        "template": 2,
        "example": 3,
        "rubric": 1,
        "anti_pattern": 1,
        "workflow": 1,
        "raw_chunk": 2,
    },
    "storyboard_agent": {
        "principle": 5,
        "template": 2,
        "example": 3,
        "rubric": 1,
        "anti_pattern": 1,
        "workflow": 1,
        "raw_chunk": 2,
    },
}


def _quota_for(request: SkillPackRequest) -> Dict[str, int]:
    return dict(AGENT_QUOTAS.get(request.agent_type, AGENT_QUOTAS["script_agent"]))


def _append(grouped: Dict[str, List[SkillArtifact]], artifact: SkillArtifact) -> None:
    grouped.setdefault(artifact.artifact_type, []).append(artifact)


def _apply_graph_evaluation_signals(
    *,
    store: SkillMemoryStore,
    ranked: List[RankedArtifact],
) -> tuple[List[RankedArtifact], Dict[str, Dict[str, object]]]:
    graph_db = Path(store.db_path).parent / "graph_memory.sqlite3"
    if not graph_db.exists():
        return ranked, {}

    graph_store = GraphMemoryStore(graph_db)
    explanations: Dict[str, Dict[str, object]] = {}
    adjusted: List[RankedArtifact] = []
    for row in ranked:
        signal = read_skill_graph_signal(graph_store, artifact_id=row.artifact.artifact_id)
        uses = int(signal.get("uses_count") or 0)
        reinforces = int(signal.get("reinforce_count") or 0)
        critiques = int(signal.get("critique_count") or 0)
        supports = int(signal.get("support_count") or 0)
        evidence_volume = uses + reinforces + critiques

        # Sparse history should not dominate baseline ranking.
        if evidence_volume < 2:
            boost = 0.0
            penalty = 0.0
            signal_mode = "fallback_sparse_history"
        else:
            boost = min(5.0, (reinforces * 1.6) + (uses * 0.4) + min(1.0, supports * 0.2))
            penalty = min(4.5, critiques * 1.5)
            signal_mode = "signal_applied"

        net = max(-4.5, min(5.0, boost - penalty))
        if signal_mode == "fallback_sparse_history":
            net = 0.0

        reasons = list(row.reasons)
        if net > 0:
            reasons.append("graph_eval_boost")
        elif net < 0:
            reasons.append("graph_eval_penalty")
        elif signal_mode == "fallback_sparse_history":
            reasons.append("graph_eval_fallback")

        adjusted_score = float(row.score) + float(net)
        adjusted.append(RankedArtifact(artifact=row.artifact, score=adjusted_score, reasons=reasons))
        explanations[row.artifact.artifact_id] = {
            "base_score": float(row.score),
            "adjusted_score": float(adjusted_score),
            "adjustment": float(net),
            "signal_mode": signal_mode,
            "graph_signal": {
                "uses_count": uses,
                "reinforce_count": reinforces,
                "critique_count": critiques,
                "support_count": supports,
            },
            "reasons": reasons,
        }

    adjusted.sort(key=lambda r: (-r.score, r.artifact.artifact_id))
    return adjusted, explanations


def build_skill_pack(*, store: SkillMemoryStore, request: SkillPackRequest) -> SkillPack:
    ranked = retrieve_skill_artifacts(store=store, request=request, include_raw=True)
    ranked, rank_explanations = _apply_graph_evaluation_signals(store=store, ranked=ranked)
    quotas = _quota_for(request)
    grouped: Dict[str, List[SkillArtifact]] = {}
    selected_ids: List[str] = []
    used = set()
    max_items = max(1, int(request.max_items))
    typed_selected = 0

    for art_type in TYPE_ORDER:
        quota = quotas.get(art_type, 0)
        if quota <= 0:
            continue
        for candidate in ranked:
            art = candidate.artifact
            if art.artifact_type != art_type:
                continue
            if art.artifact_id in used:
                continue
            if len(grouped.get(art_type, [])) >= quota:
                continue
            if len(selected_ids) >= max_items:
                break
            if art_type == "raw_chunk" and typed_selected >= 6:
                continue
            _append(grouped, art)
            used.add(art.artifact_id)
            selected_ids.append(art.artifact_id)
            if art_type != "raw_chunk":
                typed_selected += 1
        if len(selected_ids) >= max_items:
            break

    warnings: List[str] = []
    if typed_selected < 4:
        warnings.append("Skill pack coverage is thin; insufficient distilled artifacts for this request.")
    for required in ("principle", "template", "example"):
        if not grouped.get(required):
            warnings.append(f"Missing preferred artifact type: {required}")

    return SkillPack(
        query=request.query,
        agent_type=request.agent_type,
        task_type=request.task_type,
        channel_type=request.channel_type,
        language=request.language,
        style_profile=request.style_profile,
        selected_artifact_ids=selected_ids,
        grouped_artifacts=grouped,
        warnings=warnings,
        selection_explanations={k: rank_explanations.get(k, {}) for k in selected_ids},
    )
