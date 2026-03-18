from __future__ import annotations

import asyncio
from pathlib import Path


class IngestionJobManager:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._lock = asyncio.Lock()

    async def ingest(self, path: Path) -> list[str]:
        async with self._lock:
            return self.pipeline.ingest_path(path)
