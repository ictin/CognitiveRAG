from fastapi import APIRouter, Depends

from CognitiveRAG.api.dependencies import get_services
from CognitiveRAG.core.lifecycle import Services
from CognitiveRAG.schemas.api import (
    UpsertEventRequest,
    UpsertProfileFactRequest,
    UpsertReasoningRequest,
    UpsertTaskRequest,
)

router = APIRouter()


@router.post("/events")
async def upsert_event(payload: UpsertEventRequest, services: Services = Depends(get_services)) -> dict:
    services.episodic_store.upsert(payload.event)
    return {"ok": True}


@router.post("/profile")
async def upsert_profile(payload: UpsertProfileFactRequest, services: Services = Depends(get_services)) -> dict:
    services.profile_store.upsert(payload.fact)
    return {"ok": True}


@router.post("/tasks")
async def upsert_task(payload: UpsertTaskRequest, services: Services = Depends(get_services)) -> dict:
    services.task_store.upsert(payload.task)
    return {"ok": True}


@router.post("/reasoning")
async def upsert_reasoning(payload: UpsertReasoningRequest, services: Services = Depends(get_services)) -> dict:
    services.reasoning_store.upsert(payload.pattern)
    return {"ok": True}
