from pathlib import Path

from fastapi import APIRouter, Depends

from CognitiveRAG.api.dependencies import get_services
from CognitiveRAG.core.lifecycle import Services
from CognitiveRAG.schemas.api import IngestPathRequest, IngestResponse

router = APIRouter()


@router.post("", response_model=IngestResponse)
async def ingest(payload: IngestPathRequest, services: Services = Depends(get_services)) -> IngestResponse:
    document_ids = await services.job_manager.ingest(Path(payload.path))
    return IngestResponse(
        accepted=True,
        message=f"Ingested {len(document_ids)} document(s).",
        document_ids=document_ids,
    )
