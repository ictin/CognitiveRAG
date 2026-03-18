from __future__ import annotations

from CognitiveRAG.llm.prompts import SYNTHESIZER_SYSTEM, synthesizer_user_prompt
from CognitiveRAG.schemas.agent import AnswerDraft
from CognitiveRAG.schemas.retrieval import RetrievalBundle


class SynthesizerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def run(self, query: str, retrieval: RetrievalBundle) -> AnswerDraft:
        # use only top chunk(s) to keep prompt size reasonable for Ollama
        top_chunks = retrieval.chunks[:2]
        context = "\n\n".join(
            f"[{chunk.source_type}:{chunk.chunk_id}] {chunk.text}"
            for chunk in top_chunks
        )
        answer = await self.llm_client.ainvoke_text(
            system_prompt=SYNTHESIZER_SYSTEM,
            user_prompt=synthesizer_user_prompt(query, context),
        )
        return AnswerDraft(
            answer=answer,
            citations=[chunk.chunk_id for chunk in retrieval.chunks],
            confidence=0.5,
        )
