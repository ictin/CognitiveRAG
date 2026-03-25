from CognitiveRAG.schemas.api import QueryResponse
from CognitiveRAG.schemas.agent import OrchestrationTrace, Plan, Critique
from CognitiveRAG.schemas.memory import ArtifactExact, DerivedSummary, build_context_block


def test_query_response_context_block_exact_derived_split():
    exact_item = ArtifactExact(
        item_id="artifact-1",
        source="query",
        content="exact artifact text",
        metadata={"kind": "artifact"},
        provenance={"source": "unit-test"},
    )
    derived_item = DerivedSummary(
        item_id="summary-1",
        source="query",
        content="derived summary text",
        summary="derived summary text",
        metadata={"kind": "summary"},
        provenance={"source": "unit-test"},
    )

    context_block = build_context_block(
        block_id="ctx_test_1",
        session_id="session-1",
        project="project-1",
        task_id="task-1",
        exact_items=[exact_item],
        derived_items=[derived_item],
        provenance={"query": "hello"},
    )

    response = QueryResponse(
        answer="ok",
        trace=OrchestrationTrace(
            plan=Plan(objective="test", steps=[]),
            critique=Critique(approved=True),
        ),
        context_block=context_block,
    )

    assert response.context_block is not None
    assert len(response.context_block.exact_items) == 1
    assert len(response.context_block.derived_items) == 1

    exact_out = response.context_block.exact_items[0]
    derived_out = response.context_block.derived_items[0]

    assert exact_out.exactness == "exact"
    assert exact_out.summarizable is False
    assert exact_out.item_type == "artifact_exact"

    assert derived_out.exactness == "derived"
    assert derived_out.summarizable is False
    assert derived_out.item_type == "derived_summary"
    assert response.context_block.to_prompt_payload()["provenance"]["query"] == "hello"
