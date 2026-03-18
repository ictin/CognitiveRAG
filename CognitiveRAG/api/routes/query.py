from fastapi import APIRouter, Depends

from CognitiveRAG.api.dependencies import get_services
from CognitiveRAG.core.lifecycle import Services
from CognitiveRAG.schemas.api import QueryRequest, QueryResponse

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query(payload: QueryRequest, services: Services = Depends(get_services)) -> QueryResponse:
    logger.info("LOG: QUERY LEXICAL payload.lexical_only=%s retrieval_mode=%s", payload.lexical_only, payload.retrieval_mode)
    return await services.orchestrator.run(payload.query, lexical_only=payload.lexical_only, retrieval_mode=payload.retrieval_mode)
