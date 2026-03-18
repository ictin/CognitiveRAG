from __future__ import annotations

from dataclasses import dataclass

from CognitiveRAG.core.settings import Settings


@dataclass
class RetrievalPlan:
    intent: str
    use_internal: bool = True
    use_episodic: bool = True
    use_graph: bool = True
    use_web: bool = False


class RetrievalRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def route(self, query: str) -> RetrievalPlan:
        lowered = query.lower()
        use_web = self.settings.retrieval.web_search_enabled and any(
            token in lowered for token in ("latest", "current", "today", "news")
        )
        return RetrievalPlan(
            intent="general",
            use_internal=True,
            use_episodic=True,
            use_graph=True,
            use_web=use_web,
        )
