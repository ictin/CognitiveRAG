from __future__ import annotations

from CognitiveRAG.schemas.retrieval import RetrievedChunk


class WebSearchClient:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    async def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.enabled:
            return []
        # TODO: integrate trusted search provider / MCP / local search stack
        return []
