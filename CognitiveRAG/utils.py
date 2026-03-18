# Lightweight utils shim to avoid heavy nltk downloads during import
from typing import List, Any

def chunk_text_simple(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


def tokenize(text: str) -> List[str]:
    return text.lower().split()

# Provide minimal Document-like structure to satisfy existing imports
class Document:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Keep API compatibility for earlier functions
def ingest_files_from_folder(folder_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    from pathlib import Path
    import json
    docs: List[Document] = []
    for filepath in Path(folder_path).glob("*"):
        try:
            if filepath.suffix.lower() in {".txt", ".md"}:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            elif filepath.suffix.lower() == ".json":
                payload = json.loads(filepath.read_text(encoding='utf-8', errors='ignore'))
                content = str(payload)
            else:
                continue
            for i, chunk in enumerate(chunk_text_simple(content, chunk_size, chunk_overlap)):
                docs.append(Document(page_content=chunk, metadata={"source": filepath.name, "chunk_index": i}))
        except Exception:
            continue
    return docs
