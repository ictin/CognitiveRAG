from fastapi import FastAPI

from CognitiveRAG.api.routes.admin import router as admin_router
from CognitiveRAG.api.routes.health import router as health_router
from CognitiveRAG.api.routes.ingest import router as ingest_router
from CognitiveRAG.api.routes.memory import router as memory_router
from CognitiveRAG.api.routes.query import router as query_router
from CognitiveRAG.core.lifecycle import lifespan
from CognitiveRAG.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="CognitiveRAG",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.include_router(health_router, tags=["health"])
    app.include_router(query_router, prefix="/query", tags=["query"])
    app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
    app.include_router(memory_router, prefix="/memory", tags=["memory"])
    app.include_router(admin_router, prefix="/admin", tags=["admin"])

    return app


app = create_app()
