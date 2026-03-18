from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from CognitiveRAG.core.settings import settings, Settings
# lightweight placeholders to avoid import errors during scaffold
from CognitiveRAG.stores.metadata_store import MetadataStore
from CognitiveRAG.stores.vector_store import VectorStore
from CognitiveRAG.stores.lexical_store import LexicalStore
from CognitiveRAG.stores.graph_store import GraphStore
from CognitiveRAG.memory.episodic_store import EpisodicStore
from CognitiveRAG.memory.profile_store import ProfileStore
from CognitiveRAG.memory.reasoning_store import ReasoningStore
from CognitiveRAG.memory.task_store import TaskStore
from CognitiveRAG.retrieval.hybrid import HybridRetriever
from CognitiveRAG.retrieval.router import RetrievalRouter
from CognitiveRAG.retrieval.web import WebSearchClient
from CognitiveRAG.ingest.pipeline import IngestionPipeline
from CognitiveRAG.ingest.jobs import IngestionJobManager
from CognitiveRAG.agents.orchestrator import Orchestrator
from CognitiveRAG.llm.factory import build_llm_clients


@dataclass
class Services:
    settings: Settings
    metadata_store: MetadataStore
    vector_store: VectorStore
    lexical_store: LexicalStore
    graph_store: GraphStore
    episodic_store: EpisodicStore
    profile_store: ProfileStore
    task_store: TaskStore
    reasoning_store: ReasoningStore
    web_search: WebSearchClient
    retriever: HybridRetriever
    router: RetrievalRouter
    ingest_pipeline: IngestionPipeline
    job_manager: IngestionJobManager
    orchestrator: Orchestrator


def detect_ollama_base() -> tuple[str, bool, str]:
    import requests, subprocess
    candidates = [
        ("http://ollamahost", 80),
        ("http://winhost:11434", 11434),
        ("http://127.0.0.1:11434", 11434),
    ]
    api_path = "/api"
    for base, port in candidates:
        try:
            r = requests.get(f"{base}{api_path}/tags", timeout=2)
            if r.status_code == 200:
                return base, True, "candidate"
        except Exception:
            continue
    # fallback to gateway ip
    try:
        out = subprocess.check_output("ip route show | grep -i default | awk '{ print $3 }'", shell=True, text=True).strip()
        if out:
            base = f"http://{out}:11434"
            r = requests.get(f"{base}{api_path}/tags", timeout=2)
            if r.status_code == 200:
                return base, True, "gateway"
    except Exception:
        pass
    return settings.llm.ollama_base_url, False, "default"


def build_services() -> Services:
    s = settings
    s.store.data_dir.mkdir(parents=True, exist_ok=True)
    s.store.source_documents_dir.mkdir(parents=True, exist_ok=True)

    import logging
    logging.getLogger().info('STARTUP_STEP: before MetadataStore')
    metadata_store = MetadataStore(s.store.metadata_db_path)
    logging.getLogger().info('STARTUP_STEP: after MetadataStore')

    logging.getLogger().info('STARTUP_STEP: before VectorStore')
    vector_store = VectorStore(s.store.vector_store_path, embedding_model=s.store.chroma_embedding_model, backing_impl=s.store.chroma_backing_impl)
    logging.getLogger().info('STARTUP_STEP: after VectorStore')

    logging.getLogger().info('STARTUP_STEP: before LexicalStore')
    lexical_store = LexicalStore()
    logging.getLogger().info('STARTUP_STEP: after LexicalStore')

    logging.getLogger().info('STARTUP_STEP: before GraphStore')
    graph_store = GraphStore(s.store.graph_db_path)
    logging.getLogger().info('STARTUP_STEP: after GraphStore')

    logging.getLogger().info('STARTUP_STEP: before Episodic/Profile/Task/Reasoning stores')
    episodic_store = EpisodicStore(s.store.episodic_db_path)
    profile_store = ProfileStore(s.store.profile_db_path)
    task_store = TaskStore(s.store.task_db_path)
    reasoning_store = ReasoningStore(s.store.reasoning_db_path)
    logging.getLogger().info('STARTUP_STEP: after Episodic/Profile/Task/Reasoning stores')

    # Use configured Ollama base from settings (no blocking discovery in startup)
    logging.getLogger().info('STARTUP_STEP: using configured ollama_base = %s api_path=%s', s.llm.ollama_base_url, s.llm.ollama_api_path)

    logging.getLogger().info('STARTUP_STEP: before build_llm_clients')
    llm_clients = build_llm_clients(s)
    logging.getLogger().info('STARTUP_STEP: after build_llm_clients')

    logging.getLogger().info('STARTUP_STEP: before WebSearchClient')
    web_search = WebSearchClient(enabled=s.retrieval.web_search_enabled)
    logging.getLogger().info('STARTUP_STEP: after WebSearchClient')

    # Create retriever instance and inject shared store instances so retriever reuses them
    from CognitiveRAG.retriever import set_memory_stores as _set_memory_stores

    retriever = HybridRetriever(
        settings=s,
        metadata_store=metadata_store,
        vector_store=vector_store,
        lexical_store=lexical_store,
        graph_store=graph_store,
        episodic_store=episodic_store,
        web_search=web_search,
        task_store=task_store,
        profile_store=profile_store,
        reasoning_store=reasoning_store,
    )

    # provide injected store instances to the lightweight retriever module for reuse
    try:
        _set_memory_stores(task_store=task_store, profile_store=profile_store, reasoning_store=reasoning_store, episodic_store=episodic_store)
    except Exception:
        pass
    router = RetrievalRouter(settings=s)
    ingest_pipeline = IngestionPipeline(
        settings=s,
        metadata_store=metadata_store,
        vector_store=vector_store,
        lexical_store=lexical_store,
        graph_store=graph_store,
    )
    job_manager = IngestionJobManager(ingest_pipeline)
    orchestrator = Orchestrator(
        settings=s,
        llm_clients=llm_clients,
        router=router,
        retriever=retriever,
        episodic_store=episodic_store,
        task_store=task_store,
        reasoning_store=reasoning_store,
    )

    return Services(
        settings=s,
        metadata_store=metadata_store,
        vector_store=vector_store,
        lexical_store=lexical_store,
        graph_store=graph_store,
        episodic_store=episodic_store,
        profile_store=profile_store,
        task_store=task_store,
        reasoning_store=reasoning_store,
        web_search=web_search,
        retriever=retriever,
        router=router,
        ingest_pipeline=ingest_pipeline,
        job_manager=job_manager,
        orchestrator=orchestrator,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = build_services()
    yield
