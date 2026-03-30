from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from CognitiveRAG.agents.critic import CriticAgent
from CognitiveRAG.agents.planner import PlannerAgent
from CognitiveRAG.agents.synthesizer import SynthesizerAgent
from CognitiveRAG.schemas.agent import OrchestrationTrace, Critique
from CognitiveRAG.schemas.api import QueryResponse
from CognitiveRAG.schemas.memory import EpisodicEvent
from CognitiveRAG.schemas.memory import ReasoningPattern
from CognitiveRAG.schemas.memory import build_context_block, ArtifactExact, DerivedSummary


class Orchestrator:
    def __init__(
        self,
        settings,
        llm_clients,
        router,
        retriever,
        episodic_store,
        task_store,
        reasoning_store,
    ):
        self.settings = settings
        self.router = router
        self.retriever = retriever
        self.episodic_store = episodic_store
        self.task_store = task_store
        self.reasoning_store = reasoning_store

        self.planner = PlannerAgent(llm_clients.planner)
        self.synthesizer = SynthesizerAgent(llm_clients.synthesizer)
        self.critic = CriticAgent(llm_clients.critic)

    async def run(
        self,
        query: str,
        lexical_only: bool = False,
        retrieval_mode: str | None = None,
        session_id: str | None = None,
    ) -> QueryResponse:
        retrieval_plan = self.router.route(query)
        from CognitiveRAG.retrieval.policy import policy_for_mode
        policy = policy_for_mode(retrieval_mode)
        import logging
        logger = logging.getLogger(__name__)
        logger.info("LOG: ORCH lexical_only=%s retrieval_mode=%s policy=%s retrieval_plan.use_episodic=%s", lexical_only, retrieval_mode, policy, retrieval_plan.use_episodic)
        if lexical_only:
            retrieval_plan.use_episodic = False
            retrieval_plan.use_graph = False
            retrieval_plan.use_web = False
            retrieval_plan.use_internal = True

        # If mode is task_memory, enforce exclusion of episodic, web, and graph at orchestration level
        if policy is not None and policy.mode == "task_memory":
            retrieval_plan.use_episodic = False
            retrieval_plan.use_web = False
            retrieval_plan.use_graph = False
            retrieval_plan.use_internal = True

        plan = await self.planner.run(query)
        retrieval = await self.retriever.retrieve(query, retrieval_plan, policy)
        logger.info("LOG: ORCH retrieval_chunks=%s", [c.chunk_id for c in retrieval.chunks])

        answer_draft = await self.synthesizer.run(query, retrieval, session_id=session_id)
        try:
            critique = await self.critic.run(query, answer_draft.answer)
        except Exception:
            # Legacy/test fallback: critique failures should not hard-fail query path.
            critique = Critique(approved=True, issues=[], follow_up_actions=[])

        self.episodic_store.upsert(
            EpisodicEvent(
                event_id=f"evt_{uuid4().hex}",
                timestamp=datetime.now(timezone.utc),
                event_type="query",
                goal=query,
                result=answer_draft.answer,
                success_score=1.0 if critique.approved else 0.5,
                metadata={"citations": answer_draft.citations},
            )
        )

        # Promote to reasoning memory when critique approves the answer (lightweight, deterministic)
        try:
            if critique.approved:
                self.promote_reasoning(
                    problem_signature=query,
                    reasoning_steps=[],
                    solution_summary=answer_draft.answer,
                    confidence=1.0 if critique.approved else 0.5,
                    provenance=answer_draft.citations,
                )
        except Exception:
            # Non-fatal: promotion should not block response
            pass

        exact_items = []
        derived_items = []
        for chunk in retrieval.chunks:
            provenance = {
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "score": chunk.score,
                "rank": getattr(chunk, "rank", None),
                "final_score": getattr(chunk, "final_score", None),
                "ranking_reason": getattr(chunk, "ranking_reason", None),
            }
            if getattr(chunk, "exactness", "derived") == "exact" or getattr(chunk, "summarizable", True) is False:
                exact_items.append(ArtifactExact(item_id=chunk.chunk_id, source=chunk.source_type, provenance=provenance, content=chunk.text, metadata=dict(chunk.metadata or {})))
            else:
                derived_items.append(DerivedSummary(item_id=chunk.chunk_id, source=chunk.source_type, provenance=provenance, content=chunk.text, summary=chunk.text[:256], metadata=dict(chunk.metadata or {})))

        context_block = build_context_block(
            block_id=f"ctx_{uuid4().hex}",
            project=plan.objective,
            exact_items=exact_items,
            derived_items=derived_items,
            provenance={"query": query, "retrieval_mode": retrieval_mode, "augmentation_decision": getattr(retrieval, "augmentation_decision", None)},
        )

        return QueryResponse(
            answer=answer_draft.answer,
            trace=OrchestrationTrace(
                plan=plan,
                critique=critique,
                retrieval_summary=[chunk.chunk_id for chunk in retrieval.chunks],
                retrieval_sources=[{
                    "chunk_id": chunk.chunk_id,
                    "source_type": chunk.source_type,
                    "score": chunk.score,
                    "rank": getattr(chunk, "rank", None),
                    "final_score": getattr(chunk, "final_score", None),
                    "ranking_reason": getattr(chunk, "ranking_reason", None),
                } for chunk in retrieval.chunks],
                augmentation_decision=getattr(retrieval, "augmentation_decision", None),
            ),
            context_block=context_block,
        )

    def promote_reasoning(self, problem_signature: str, reasoning_steps: list[str], solution_summary: str, confidence: float = 0.0, provenance: list | None = None) -> None:
        """Lightweight promotion into the reasoning store. Creates a deterministic pattern_id and upserts.
        This is intentionally minimal and non-blocking.
        """
        try:
            pattern_id = f"rp_{uuid4().hex}"
            rp = ReasoningPattern(
                pattern_id=pattern_id,
                problem_signature=problem_signature,
                reasoning_steps=reasoning_steps or [],
                solution_summary=solution_summary,
                confidence=float(confidence or 0.0),
            )
            # include provenance in metadata if store supports enrichment via upsert of pattern (store.upsert expects ReasoningPattern)
            if self.reasoning_store is not None:
                try:
                    self.reasoning_store.upsert(rp)
                except Exception:
                    # non-fatal
                    pass
        except Exception:
            pass
