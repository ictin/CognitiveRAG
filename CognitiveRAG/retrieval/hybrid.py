from __future__ import annotations

from CognitiveRAG.core.settings import Settings
from CognitiveRAG.retrieval.rerank import rerank_chunks
from CognitiveRAG.schemas.retrieval import RetrievalBundle, RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        metadata_store,
        vector_store,
        lexical_store,
        graph_store,
        episodic_store,
        web_search,
        task_store=None,
        profile_store=None,
        reasoning_store=None,
    ):
        self.settings = settings
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.lexical_store = lexical_store
        self.graph_store = graph_store
        self.episodic_store = episodic_store
        self.web_search = web_search
        self.task_store = task_store
        self.profile_store = profile_store
        self.reasoning_store = reasoning_store

    async def retrieve(self, query: str, plan, policy=None) -> RetrievalBundle:
        chunks: list[RetrievedChunk] = []

        # Determine vector where filter based on policy
        vector_where = None
        if policy is not None and not getattr(policy, 'allow_episodic', True):
            # policy that disallows episodic should restrict to stored document chunks
            vector_where = {"source_type": "document"}

        if plan.use_internal:
            chunks.extend(self.vector_store.query(query, top_k=self.settings.retrieval.vector_top_k, where=vector_where))
            chunks.extend(self.lexical_store.query(query, top_k=self.settings.retrieval.bm25_top_k))

        if plan.use_graph:
            chunks.extend(self.graph_store.query(query, top_k=self.settings.retrieval.graph_top_k))

        # Respect policy: skip episodic retrieval if not allowed
        if plan.use_episodic and (policy is None or getattr(policy, 'allow_episodic', True)):
            chunks.extend(self.episodic_store.query(query, top_k=self.settings.retrieval.episodic_top_k))

        # Web augmentation: only perform web search when plan requests it AND policy allows it
        # (policy.allow_web defaults to plan.use_web if not provided)
        do_web_search = False
        if getattr(plan, 'use_web', False):
            if policy is None:
                do_web_search = True
            else:
                do_web_search = bool(getattr(policy, 'allow_web', getattr(plan, 'use_web', False)))

        if do_web_search:
            chunks.extend(await self.web_search.search(query, top_k=5))

        # If policy indicates task_memory mode, include non-episodic task/profile/reasoning sources
        try:
            if policy is not None and getattr(policy, 'mode', None) == 'task_memory':
                # query task/profile/reasoning stores for relevant items and merge
                tresults: list[dict] = []
                presults: list[dict] = []
                rresults: list[dict] = []
                if getattr(policy, 'allow_task', True) and self.task_store is not None:
                    try:
                        tresults = self.task_store.query(query, top_k=3)
                    except Exception:
                        tresults = []
                if getattr(policy, 'allow_profile', True) and self.profile_store is not None:
                    try:
                        presults = self.profile_store.query(query, top_k=3)
                    except Exception:
                        presults = []
                if getattr(policy, 'allow_reasoning', True) and self.reasoning_store is not None:
                    try:
                        rresults = self.reasoning_store.query(query, top_k=3)
                    except Exception:
                        rresults = []

                # convert to RetrievedChunk and extend
                for d in tresults + presults + rresults:
                    try:
                        chunks.append(RetrievedChunk(**d))
                    except Exception:
                        continue
        except Exception:
            # non-fatal: don't break retrieval if stores fail
            pass

        # final rerank and bundle
        merged = rerank_chunks(chunks, max_items=self.settings.retrieval.max_context_chunks)

        # Source balancing: cap web results to at most 2 and ensure at least one non-web survives when available
        web_chunks = [c for c in merged if getattr(c, 'source_type', None) == 'web']
        non_web_chunks = [c for c in merged if getattr(c, 'source_type', None) != 'web']

        # cap web to max 2
        max_web = 2
        web_kept = web_chunks[:max_web]

        # if there are non-web chunks available, ensure at least one non-web is present
        if non_web_chunks:
            # take at least one non-web, then fill up to max items with remaining non-web + web_kept preserving order from merged
            remaining_slots = max(1, self.settings.retrieval.max_context_chunks)  # ensure at least one slot
            selected = []
            for c in merged:
                if getattr(c, 'source_type', None) == 'web':
                    if c in web_kept and len(selected) < self.settings.retrieval.max_context_chunks:
                        selected.append(c)
                else:
                    if len(selected) < self.settings.retrieval.max_context_chunks:
                        selected.append(c)
        else:
            # no non-web available, just take web_kept up to max items
            selected = web_kept[: self.settings.retrieval.max_context_chunks]

        # attach deterministic ranking/debug info derived from selected order
        final_chunks: list[RetrievedChunk] = []
        for idx, c in enumerate(selected, start=1):
            c.rank = idx
            c.final_score = c.score if c.score is not None else 0.0
            c.ranking_reason = f"source={c.source_type};score={c.score}"
            final_chunks.append(c)

        # build augmentation decision (considered/allowed/used)
        considered = bool(getattr(plan, 'use_web', False))
        allowed = bool(getattr(policy, 'allow_web', getattr(plan, 'use_web', False)))
        used = any(getattr(c, 'source_type', None) == 'web' for c in final_chunks)

        if considered and not allowed:
            reason = "web requested by plan but blocked by policy"
        elif considered and allowed and used:
            reason = "web used because plan requested it and policy allowed it"
        elif considered and allowed and not used:
            reason = "web allowed but search returned no usable results"
        else:
            reason = "web not requested by plan"

        return RetrievalBundle(
            query=query,
            intent=plan.intent,
            chunks=final_chunks,
            augmentation_decision={
                "considered": considered,
                "allowed": allowed,
                "used": used,
                "reason": reason,
            },
        )