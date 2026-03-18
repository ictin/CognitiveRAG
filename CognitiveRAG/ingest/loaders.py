from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDocument:
    document_id: str
    source_path: str
    content: str
    content_hash: str
    metadata: dict


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_path(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        content = path.read_text(encoding="utf-8")
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        content = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    content_hash = _sha256_text(content)
    document_id = f"doc_{content_hash[:16]}"
    return LoadedDocument(
        document_id=document_id,
        source_path=str(path),
        content=content,
        content_hash=content_hash,
        metadata={"source_path": str(path), "suffix": suffix},
    )
