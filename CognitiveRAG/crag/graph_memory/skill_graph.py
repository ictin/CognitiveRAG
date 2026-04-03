from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from .schemas import GraphEdge, GraphNode, GraphRelationType, stable_edge_id, stable_node_id
from .store import GraphMemoryStore

if TYPE_CHECKING:
    from CognitiveRAG.crag.skill_memory.evaluation_schema import SkillEvaluationCase
    from CognitiveRAG.crag.skill_memory.execution_schema import SkillExecutionCase
    from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _skill_artifact_node(artifact: SkillArtifact, *, now_iso: str) -> GraphNode:
    return GraphNode(
        node_id=stable_node_id("skill_artifact", artifact.artifact_id),
        node_type="skill_artifact",
        label=artifact.title or artifact.canonical_text[:140],
        properties={
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type,
            "namespace": artifact.namespace,
            "normalized_key": artifact.normalized_key,
            "confidence": float(artifact.confidence),
            "tags": list(artifact.tags),
        },
        provenance={
            "source_class": "skill_memory_artifact",
            "source_refs": [ref.model_dump() for ref in artifact.source_refs],
        },
        created_at=now_iso,
        updated_at=now_iso,
    )


def _category_node(category: str, *, now_iso: str) -> GraphNode:
    return GraphNode(
        node_id=stable_node_id("category", category),
        node_type="category",
        label=category,
        properties={"category": category},
        provenance={"source": "skill_graph", "kind": "deterministic_skill_category"},
        created_at=now_iso,
        updated_at=now_iso,
    )


def _source_ref_node(artifact: SkillArtifact, source_ref: Dict[str, Any], *, now_iso: str) -> GraphNode:
    source_key = str(source_ref.get("chunk_id") or source_ref.get("source_path") or artifact.artifact_id)
    return GraphNode(
        node_id=stable_node_id("skill_source_ref", source_key),
        node_type="skill_source_ref",
        label=source_key,
        properties={k: v for k, v in source_ref.items()},
        provenance={"source_class": "skill_artifact_source_ref", "artifact_id": artifact.artifact_id},
        created_at=now_iso,
        updated_at=now_iso,
    )


def record_skill_artifact_graph_links(
    store: GraphMemoryStore,
    *,
    artifact: SkillArtifact,
    now_iso: str | None = None,
) -> List[GraphEdge]:
    now = now_iso or _now_iso()
    artifact_node = _skill_artifact_node(artifact, now_iso=now)

    nodes: List[GraphNode] = [artifact_node]
    edges: List[GraphEdge] = []

    categories = [f"skill_type:{artifact.artifact_type}", f"skill_namespace:{artifact.namespace}"]
    for category in categories:
        cat_node = _category_node(category, now_iso=now)
        nodes.append(cat_node)
        edges.append(
            GraphEdge(
                edge_id=stable_edge_id(artifact_node.node_id, GraphRelationType.BELONGS_TO_CATEGORY, cat_node.node_id),
                source_node_id=artifact_node.node_id,
                relation_type=GraphRelationType.BELONGS_TO_CATEGORY,
                target_node_id=cat_node.node_id,
                properties={"category": category},
                provenance={
                    "reason": "skill_artifact_deterministic_category_link",
                    "artifact_id": artifact.artifact_id,
                },
                created_at=now,
                updated_at=now,
            )
        )

    for source_ref in [ref.model_dump() for ref in artifact.source_refs]:
        src = _source_ref_node(artifact, source_ref, now_iso=now)
        nodes.append(src)
        edges.append(
            GraphEdge(
                edge_id=stable_edge_id(artifact_node.node_id, GraphRelationType.SUPPORTED_BY, src.node_id),
                source_node_id=artifact_node.node_id,
                relation_type=GraphRelationType.SUPPORTED_BY,
                target_node_id=src.node_id,
                properties={"artifact_id": artifact.artifact_id},
                provenance={
                    "reason": "skill_artifact_supported_by_source_ref",
                    "artifact_id": artifact.artifact_id,
                },
                created_at=now,
                updated_at=now,
            )
        )

    store.upsert_nodes(nodes)
    store.upsert_edges(edges)
    return edges


def record_execution_case_graph_links(
    store: GraphMemoryStore,
    *,
    case: SkillExecutionCase,
    now_iso: str | None = None,
) -> List[GraphEdge]:
    now = now_iso or _now_iso()
    exec_node = GraphNode(
        node_id=stable_node_id("execution_case", case.execution_case_id),
        node_type="execution_case",
        label=case.execution_case_id,
        properties={
            "execution_case_id": case.execution_case_id,
            "agent_type": case.agent_type,
            "task_type": case.task_type,
            "channel_type": case.channel_type,
            "language": case.language,
            "success_flag": bool(case.success_flag),
            "pack_ref": case.pack_ref,
        },
        provenance={"source_class": "skill_execution_case", **dict(case.provenance.model_dump())},
        created_at=now,
        updated_at=now,
    )
    nodes = [exec_node]
    edges: List[GraphEdge] = []

    for artifact_id in sorted(set(case.selected_artifact_ids)):
        art_node = GraphNode(
            node_id=stable_node_id("skill_artifact", artifact_id),
            node_type="skill_artifact",
            label=artifact_id,
            properties={"artifact_id": artifact_id},
            provenance={"source_class": "skill_memory_artifact_ref"},
            created_at=now,
            updated_at=now,
        )
        nodes.append(art_node)
        edges.append(
            GraphEdge(
                edge_id=stable_edge_id(exec_node.node_id, GraphRelationType.USES_SKILL_ARTIFACT, art_node.node_id),
                source_node_id=exec_node.node_id,
                relation_type=GraphRelationType.USES_SKILL_ARTIFACT,
                target_node_id=art_node.node_id,
                properties={"execution_case_id": case.execution_case_id, "artifact_id": artifact_id},
                provenance={"reason": "execution_case_selected_artifact", "execution_case_id": case.execution_case_id},
                created_at=now,
                updated_at=now,
            )
        )

    if case.output_ref or case.output_text:
        output_key = case.output_ref or case.execution_case_id
        output_node = GraphNode(
            node_id=stable_node_id("execution_output", output_key),
            node_type="execution_output",
            label=(case.output_ref or case.output_text[:120]),
            properties={"output_ref": case.output_ref, "output_text_preview": case.output_text[:200]},
            provenance={"source_class": "skill_execution_output", "execution_case_id": case.execution_case_id},
            created_at=now,
            updated_at=now,
        )
        nodes.append(output_node)
        edges.append(
            GraphEdge(
                edge_id=stable_edge_id(exec_node.node_id, GraphRelationType.PRODUCED_OUTPUT, output_node.node_id),
                source_node_id=exec_node.node_id,
                relation_type=GraphRelationType.PRODUCED_OUTPUT,
                target_node_id=output_node.node_id,
                properties={"execution_case_id": case.execution_case_id},
                provenance={"reason": "execution_case_output_link", "execution_case_id": case.execution_case_id},
                created_at=now,
                updated_at=now,
            )
        )

    store.upsert_nodes(nodes)
    store.upsert_edges(edges)
    return edges


def record_evaluation_case_graph_links(
    store: GraphMemoryStore,
    *,
    case: SkillEvaluationCase,
    artifact_ids: Iterable[str],
    now_iso: str | None = None,
) -> List[GraphEdge]:
    now = now_iso or _now_iso()
    eval_node = GraphNode(
        node_id=stable_node_id("evaluation_case", case.evaluation_case_id),
        node_type="evaluation_case",
        label=case.evaluation_case_id,
        properties={
            "evaluation_case_id": case.evaluation_case_id,
            "execution_case_id": case.execution_case_id,
            "overall_score": float(case.overall_score),
            "pass_flag": bool(case.pass_flag),
            "anti_pattern_hits": list(case.anti_pattern_hits),
            "task_type": case.task_type,
        },
        provenance={"source_class": "skill_evaluation_case", **dict(case.provenance.model_dump())},
        created_at=now,
        updated_at=now,
    )
    exec_node = GraphNode(
        node_id=stable_node_id("execution_case", case.execution_case_id),
        node_type="execution_case",
        label=case.execution_case_id,
        properties={"execution_case_id": case.execution_case_id},
        provenance={"source_class": "skill_execution_case_ref"},
        created_at=now,
        updated_at=now,
    )
    nodes: List[GraphNode] = [eval_node, exec_node]
    edges: List[GraphEdge] = [
        GraphEdge(
            edge_id=stable_edge_id(eval_node.node_id, GraphRelationType.EVALUATES_EXECUTION, exec_node.node_id),
            source_node_id=eval_node.node_id,
            relation_type=GraphRelationType.EVALUATES_EXECUTION,
            target_node_id=exec_node.node_id,
            properties={"evaluation_case_id": case.evaluation_case_id, "execution_case_id": case.execution_case_id},
            provenance={"reason": "evaluation_judges_execution_case"},
            created_at=now,
            updated_at=now,
        )
    ]

    relation = (
        GraphRelationType.REINFORCES_SKILL_ARTIFACT
        if bool(case.pass_flag) and float(case.overall_score) >= 0.7
        else GraphRelationType.CRITIQUES_SKILL_ARTIFACT
    )
    reason = "evaluation_reinforces_artifact" if relation == GraphRelationType.REINFORCES_SKILL_ARTIFACT else "evaluation_critiques_artifact"
    for artifact_id in sorted(set(str(a) for a in artifact_ids if str(a))):
        art_node = GraphNode(
            node_id=stable_node_id("skill_artifact", artifact_id),
            node_type="skill_artifact",
            label=artifact_id,
            properties={"artifact_id": artifact_id},
            provenance={"source_class": "skill_memory_artifact_ref"},
            created_at=now,
            updated_at=now,
        )
        nodes.append(art_node)
        edges.append(
            GraphEdge(
                edge_id=stable_edge_id(eval_node.node_id, relation, art_node.node_id),
                source_node_id=eval_node.node_id,
                relation_type=relation,
                target_node_id=art_node.node_id,
                properties={"evaluation_case_id": case.evaluation_case_id, "artifact_id": artifact_id},
                provenance={"reason": reason, "overall_score": float(case.overall_score), "pass_flag": bool(case.pass_flag)},
                created_at=now,
                updated_at=now,
            )
        )

    store.upsert_nodes(nodes)
    store.upsert_edges(edges)
    return edges


def read_skill_graph_signal(
    store: GraphMemoryStore,
    *,
    artifact_id: str,
) -> Dict[str, int]:
    node_id = stable_node_id("skill_artifact", artifact_id)
    incoming = store.get_edges_for_node(node_id, direction="incoming")
    uses = sum(1 for e in incoming if e.relation_type == GraphRelationType.USES_SKILL_ARTIFACT)
    reinforces = sum(1 for e in incoming if e.relation_type == GraphRelationType.REINFORCES_SKILL_ARTIFACT)
    critiques = sum(1 for e in incoming if e.relation_type == GraphRelationType.CRITIQUES_SKILL_ARTIFACT)
    supports = sum(1 for e in incoming if e.relation_type == GraphRelationType.SUPPORTED_BY)
    return {
        "uses_count": int(uses),
        "reinforce_count": int(reinforces),
        "critique_count": int(critiques),
        "support_count": int(supports),
    }
