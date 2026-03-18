from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class RetrievedChunk:
    page_content: str
    metadata: Dict[str, Any]
    rank: Optional[int] = None
    final_score: Optional[float] = None
    ranking_reason: Optional[str] = None
    augmentation_decision: Dict[str, Any] = field(default_factory=dict)

    def to_document(self):
        # Lazy adapter to langchain Document if needed
        try:
            from langchain_core.documents import Document
            return Document(page_content=self.page_content, metadata=self.metadata)
        except Exception:
            return None
