from __future__ import annotations

from CognitiveRAG.llm.prompts import SYNTHESIZER_SYSTEM, synthesizer_user_prompt
from CognitiveRAG.schemas.agent import AnswerDraft
from CognitiveRAG.schemas.retrieval import RetrievalBundle


class SynthesizerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def run(self, query: str, retrieval: RetrievalBundle, session_id: str | None = None) -> AnswerDraft:
        # use only top chunk(s) to keep prompt size reasonable for Ollama
        top_chunks = retrieval.chunks[:2]
        context_parts = []
        if top_chunks:
            context_parts.append("\n\n".join(
                f"[{chunk.source_type}:{chunk.chunk_id}] {chunk.text}"
                for chunk in top_chunks
            ))

        # If a session_id is provided, try to assemble session context (fresh tail + summaries)
        if session_id:
            try:
                from CognitiveRAG.retriever import assemble_session_context
                sess_ctx = assemble_session_context(session_id, fresh_tail_count=5, budget=2000)
                # include fresh_tail messages and summaries succinctly
                fresh = sess_ctx.get('fresh_tail', [])
                sums = sess_ctx.get('summaries', [])
                if fresh:
                    fresh_text = "\n".join(m.get('text','') for m in fresh)
                    context_parts.append(f"[session_fresh_tail] {fresh_text}")
                if sums:
                    sum_text = "\n".join(s.get('summary','') for s in sums)
                    context_parts.append(f"[session_summaries] {sum_text}")
            except Exception:
                # non-fatal: fall back to retrieval-only context
                pass

        context = "\n\n".join(context_parts)
        answer = await self.llm_client.ainvoke_text(
            system_prompt=SYNTHESIZER_SYSTEM,
            user_prompt=synthesizer_user_prompt(query, context),
        )
        return AnswerDraft(
            answer=answer,
            citations=[chunk.chunk_id for chunk in retrieval.chunks],
            confidence=0.5,
        )
