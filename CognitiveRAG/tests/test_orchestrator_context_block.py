import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from CognitiveRAG.agents.orchestrator import Orchestrator
from CognitiveRAG.schemas.agent import Critique, Plan
from CognitiveRAG.schemas.memory import MemoryContextBlock
from CognitiveRAG.schemas.retrieval import RetrievedChunk


def test_orchestrator_run_emits_context_block():
    exact_chunk = RetrievedChunk(
        chunk_id="exact-1",
        text="exact text",
        source_type="document",
        exactness="exact",
        summarizable=True,
        provenance={},
    ).with_policy_defaults()
    derived_chunk = RetrievedChunk(
        chunk_id="derived-1",
        text="derived text",
        source_type="vector",
        provenance={},
    ).with_policy_defaults()

    fake_retrieval = SimpleNamespace(
        chunks=[exact_chunk, derived_chunk],
        augmentation_decision={"used": False},
    )

    dummy_router = SimpleNamespace(route=lambda query: SimpleNamespace(use_episodic=False, use_graph=False, use_web=False, use_internal=True))
    dummy_retriever = SimpleNamespace(retrieve=AsyncMock(return_value=fake_retrieval))
    dummy_store = SimpleNamespace(upsert=lambda *args, **kwargs: None)
    dummy_llm = SimpleNamespace(planner=None, synthesizer=None, critic=None)

    orch = Orchestrator(
        settings=SimpleNamespace(),
        llm_clients=dummy_llm,
        router=dummy_router,
        retriever=dummy_retriever,
        episodic_store=dummy_store,
        task_store=dummy_store,
        reasoning_store=dummy_store,
    )

    orch.planner = SimpleNamespace(run=AsyncMock(return_value=Plan(objective="test", steps=[])))
    orch.synthesizer = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(answer="ok", citations=[])))
    orch.critic = SimpleNamespace(run=AsyncMock(return_value=Critique(approved=True)))
    orch.promote_reasoning = lambda *args, **kwargs: None
    orch.episodic_store = SimpleNamespace(upsert=lambda *args, **kwargs: None)
    import CognitiveRAG.agents.orchestrator as orch_mod
    orch_mod.EpisodicEvent = lambda *args, **kwargs: SimpleNamespace()

    response = asyncio.run(orch.run("hello"))

    assert response.context_block is not None
    assert isinstance(response.context_block, MemoryContextBlock)
    assert len(response.context_block.exact_items) == 1
    assert len(response.context_block.derived_items) == 1
    assert response.context_block.exact_items[0].exactness == "exact"
    assert response.context_block.exact_items[0].summarizable is False
    assert response.context_block.derived_items[0].exactness == "derived"
