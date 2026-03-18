from fastapi import APIRouter, Depends

from CognitiveRAG.api.dependencies import get_services
from CognitiveRAG.core.lifecycle import Services

router = APIRouter()


@router.get("/stats")
async def stats(services: Services = Depends(get_services)) -> dict:
    return {
        "data_dir": str(services.settings.store.data_dir),
        "web_search_enabled": services.settings.retrieval.web_search_enabled,
    }
