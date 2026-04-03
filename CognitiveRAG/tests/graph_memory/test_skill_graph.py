from pathlib import Path

from CognitiveRAG.crag.graph_memory.schemas import GraphRelationType, stable_node_id
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.execution_schema import build_execution_case
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore
from CognitiveRAG.crag.skill_memory.schemas import SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore


def test_skill_artifacts_are_linked_into_graph_deterministically(tmp_path: Path):
    skills = SkillMemoryStore(tmp_path / "skills.sqlite3")
    artifact = build_artifact(
        artifact_type="principle",
        source_ref=SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="chunk-1"),
        canonical_text="Principle: Keep hooks specific and concrete.",
        metadata={"agent_type": "script_agent"},
    )

    skills.upsert(artifact)
    skills.upsert(artifact)

    graph = GraphMemoryStore(tmp_path / "graph_memory.sqlite3")
    artifact_node_id = stable_node_id("skill_artifact", artifact.artifact_id)
    outgoing = graph.get_edges_for_node(artifact_node_id, direction="outgoing")
    assert outgoing
    assert any(e.relation_type == GraphRelationType.BELONGS_TO_CATEGORY for e in outgoing)
    assert any(e.relation_type == GraphRelationType.SUPPORTED_BY for e in outgoing)
    assert len({e.edge_id for e in outgoing}) == len(outgoing)


def test_execution_and_evaluation_cases_link_to_artifacts_in_graph(tmp_path: Path):
    skills = SkillMemoryStore(tmp_path / "skills.sqlite3")
    exec_store = SkillExecutionStore(tmp_path / "skill_exec.sqlite3")
    eval_store = SkillEvaluationStore(tmp_path / "skill_eval.sqlite3")

    artifact = build_artifact(
        artifact_type="template",
        source_ref=SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="chunk-2"),
        canonical_text="Template: {hook} {pain} {payoff}",
        metadata={"agent_type": "script_agent"},
    )
    skills.upsert(artifact)

    execution_case = build_execution_case(
        agent_type="script_agent",
        task_type="recipe_short",
        request_text="Create a short recipe intro.",
        selected_artifact_ids=[artifact.artifact_id],
        output_ref="out:1",
        output_text="Hook then payoff.",
        success_flag=True,
    )
    exec_store.upsert_case(execution_case)

    evaluation_case = build_evaluation_case(
        execution_case_id=execution_case.execution_case_id,
        agent_type="script_agent",
        task_type="recipe_short",
        criterion_scores=[RubricCriterionScore(criterion_id="hook", label="Hook", score=5, max_score=5)],
        pass_flag=True,
    )
    eval_store.upsert_case(evaluation_case)

    graph = GraphMemoryStore(tmp_path / "graph_memory.sqlite3")
    exec_node = stable_node_id("execution_case", execution_case.execution_case_id)
    eval_node = stable_node_id("evaluation_case", evaluation_case.evaluation_case_id)
    artifact_node = stable_node_id("skill_artifact", artifact.artifact_id)

    exec_edges = graph.get_edges_for_node(exec_node, direction="outgoing")
    assert any(e.relation_type == GraphRelationType.USES_SKILL_ARTIFACT and e.target_node_id == artifact_node for e in exec_edges)
    assert any(e.relation_type == GraphRelationType.PRODUCED_OUTPUT for e in exec_edges)

    eval_edges = graph.get_edges_for_node(eval_node, direction="outgoing")
    assert any(e.relation_type == GraphRelationType.EVALUATES_EXECUTION and e.target_node_id == exec_node for e in eval_edges)
    assert any(
        e.relation_type in {GraphRelationType.REINFORCES_SKILL_ARTIFACT, GraphRelationType.CRITIQUES_SKILL_ARTIFACT}
        and e.target_node_id == artifact_node
        for e in eval_edges
    )
