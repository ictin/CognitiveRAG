def test_reasoning_promotion_creates_record(tmp_path):
    from CognitiveRAG.agents.orchestrator import Orchestrator
    from CognitiveRAG.schemas.memory import ReasoningPattern
    from CognitiveRAG.memory.reasoning_store import ReasoningStore

    # create a temporary reasoning store
    dbpath = tmp_path / "reasoning.sqlite3"
    rs = ReasoningStore(dbpath)

    # create orchestrator with dummy dependencies where appropriate
    orch = Orchestrator(settings=None, llm_clients=type('X', (), {'planner':None,'synthesizer':None,'critic':None})(), router=None, retriever=None, episodic_store=None, task_store=None, reasoning_store=rs)

    # promote reasoning
    orch.promote_reasoning(problem_signature="how to test", reasoning_steps=["step1"], solution_summary="use unit tests", confidence=0.9, provenance=["cite1"])

    # query reasoning store to find the recently added pattern
    results = rs.query("unit tests", top_k=5)
    assert isinstance(results, list)
    assert any("use unit tests" in (r.get('text') or r.get('solution_summary') or '') for r in results)
