# CognitiveRAG/main_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import os

# Allow tests to skip heavy knowledge-base initialization (BERTopic, etc.) by
# setting the COGNITIVERAG_SKIP_KB environment variable. This keeps normal
# runtime behavior unchanged while allowing lightweight test runs in CI or
# constrained environments.
if os.getenv('COGNITIVERAG_SKIP_KB'):
    kb = None
else:
    from .knowledge_base import kb
import os
import json
from contextlib import asynccontextmanager
import signal
import sys
import asyncio

# --- Lifespan event handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the knowledge base on server startup
    try:
        if not kb.doc_store:
            print("Knowledge base is empty. Consider running a build.")
    except Exception as e:
        print(f"Warning: Could not check knowledge base status: {e}")
    try:
        yield
    finally:
        # Clean up resources here if needed
        print("Shutting down server and cleaning up resources...")

# Initialize FastAPI app
app = FastAPI(
    title="Cognitive RAG System",
    description="An advanced agentic RAG system for knowledge discovery.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Models ---
class IngestResponse(BaseModel):
    status: str
    message: str
    files_processed: List[str]

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- Endpoints ---

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingests one or more documents into the knowledge base.
    Saves the files to the source directory and triggers a rebuild.
    """
    try:
        from . import config
        source_dir = config.SOURCE_DOCUMENTS_DIR
        
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)

        processed_files = []
        for file in files:
            file_path = os.path.join(source_dir, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            processed_files.append(file.filename)

        # After saving files, trigger a knowledge base rebuild
        kb.build()
        return IngestResponse(
            status="success",
            message=f"Successfully ingested and rebuilt knowledge base with {len(processed_files)} files.",
            files_processed=processed_files
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build knowledge base: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Processes a user query through the multi-agent RAG pipeline.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        from .agents import Orchestrator
        orchestrator = Orchestrator()
        # pass session_id through to orchestrator.run; await the async run
        final_state = await orchestrator.run(request.query, session_id=request.session_id)
        # final_state is a QueryResponse model or dict-like; attempt to extract answer and context
        try:
            answer = final_state.answer if hasattr(final_state, 'answer') else final_state['answer']
            context_items = final_state.trace.retrieval_sources if hasattr(final_state, 'trace') else final_state.get('context', [])
            sources = sorted(list({(item.get('source') if isinstance(item, dict) else getattr(item, 'metadata', {}).get('source', 'N/A')) for item in context_items}))
        except Exception:
            # best-effort extraction fallback
            answer = final_state['answer'] if isinstance(final_state, dict) else getattr(final_state, 'answer', '')
            sources = []
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing: {str(e)}")

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

@app.get("/")
def root():
    """
    Root endpoint
    """
    return {"message": "Cognitive RAG System", "status": "running", "version": "1.0.0"}


class PromoteRequest(BaseModel):
    session_id: str


class PromoteResponse(BaseModel):
    promoted_count: int
    promoted_pattern_ids: list[str]


class AssembleContextRequest(BaseModel):
    session_id: str
    fresh_tail_count: int = 20
    budget: int = 4096
    query: str | None = None
    intent_family: str | None = None


class AssembleContextResponse(BaseModel):
    fresh_tail: list[dict]
    summaries: list[dict]
    explanation: dict | None = None
    retrieval_route: dict | None = None
    discovery_plan: dict | None = None


@app.post('/promote_session', response_model=PromoteResponse)
async def promote_session(request: PromoteRequest):
    """Operator endpoint: promote session summaries into durable reasoning memory.

    This is deliberately explicit and additive. It does not remove or modify raw session memory.
    """
    try:
        from CognitiveRAG.session_memory.promotion_bridge import promote_session_summaries
        patterns = promote_session_summaries(request.session_id, dry_run=False)
        ids = [p.pattern_id for p in patterns]
        return PromoteResponse(promoted_count=len(ids), promoted_pattern_ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {e}")

@app.post('/session_assemble_context', response_model=AssembleContextResponse)
async def session_assemble_context(request: AssembleContextRequest):
    """Assemble session context from fresh tail + summaries using context_window."""
    try:
        from CognitiveRAG.session_memory.context_window import assemble_context
        out = assemble_context(
            request.session_id,
            fresh_tail_count=int(request.fresh_tail_count),
            budget=int(request.budget),
            query=request.query,
            intent_family=request.intent_family,
        )
        return AssembleContextResponse(
            fresh_tail=list(out.get('fresh_tail') or []),
            summaries=list(out.get('summaries') or []),
            explanation=dict(out.get('explanation') or {}),
            retrieval_route=dict(out.get('retrieval_route') or {}),
            discovery_plan=dict(out.get('discovery_plan') or {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context assembly failed: {e}")


@app.post('/session_append_message')
async def session_append_message(payload: dict):
    """Minimal ingest endpoint: idempotent upsert of a session message.

    Expects JSON: {session_id, message_id, sender, text, created_at?}
    """
    session_id = payload.get('session_id')
    message_id = payload.get('message_id')
    sender = payload.get('sender')
    text = payload.get('text')
    created_at = payload.get('created_at')

    if not session_id or not message_id or not sender or text is None:
        raise HTTPException(status_code=400, detail='Missing required fields')

    # Try to use ConversationStore upsert if available
    try:
        from CognitiveRAG.session_memory.conversation_store import ConversationStore
        store = ConversationStore()
        # upsert_message expected API: upsert_message(session_id, message_id, sender, text, created_at=None)
        try:
            created = store.upsert_message(session_id, message_id, sender, text, created_at)
            status = 'inserted' if created else 'updated'
            return {'status': status}
        except Exception:
            # fallback to older API names
            try:
                store.add_message(session_id, {'message_id': message_id, 'sender': sender, 'text': text, 'created_at': created_at})
                return {'status': 'inserted'}
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: write to data/session_memory/raw_<session_id>.json . Upsert by message_id
    workdir = os.path.join(os.getcwd(), 'data', 'session_memory')
    os.makedirs(workdir, exist_ok=True)
    raw_path = os.path.join(workdir, f'raw_{session_id}.json')
    msgs = []
    if os.path.exists(raw_path):
        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                msgs = json.load(f)
        except Exception:
            msgs = []
    # find existing
    found = False
    for m in msgs:
        if str(m.get('message_id')) == str(message_id):
            m['sender'] = sender
            m['text'] = text
            m['created_at'] = created_at
            found = True
            break
    if not found:
        msgs.append({'message_id': message_id, 'sender': sender, 'text': text, 'created_at': created_at})
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(msgs, f)
    return {'status': 'inserted' if not found else 'updated'}


@app.post('/session_append_message_part')
async def session_append_message_part(payload: dict):
    """Ingest a message part (idempotent upsert).

    Expects JSON: {session_id, message_id, part_index, text, meta_json?}
    """
    session_id = payload.get('session_id')
    message_id = payload.get('message_id')
    part_index = payload.get('part_index')
    text = payload.get('text')
    meta_json = payload.get('meta_json')

    if not session_id or not message_id or part_index is None or text is None:
        raise HTTPException(status_code=400, detail='Missing required fields')

    # Try MessagePartsStore upsert if available
    try:
        from CognitiveRAG.session_memory.message_parts_store import MessagePartsStore
        mstore = MessagePartsStore()
        try:
            created = mstore.upsert_part(session_id, message_id, int(part_index), text, json.dumps(meta_json) if meta_json is not None else None)
            status = 'inserted' if created else 'updated'
            return {'status': status}
        except Exception:
            try:
                mstore.add_part(session_id, message_id, int(part_index), text, json.dumps(meta_json) if meta_json is not None else None)
                return {'status': 'inserted'}
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: write to data/session_memory/parts_<session_id>_<message_id>.json
    workdir = os.path.join(os.getcwd(), 'data', 'session_memory')
    os.makedirs(workdir, exist_ok=True)
    raw_path = os.path.join(workdir, f'parts_{session_id}_{message_id}.json')
    parts = []
    if os.path.exists(raw_path):
        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                parts = json.load(f)
        except Exception:
            parts = []
    found = False
    for p in parts:
        if int(p.get('part_index')) == int(part_index):
            p['text'] = text
            p['meta_json'] = meta_json
            found = True
            break
    if not found:
        parts.append({'part_index': int(part_index), 'text': text, 'meta_json': meta_json})
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(parts, f)
    return {'status': 'inserted' if not found else 'updated'}

@app.post('/session_upsert_context_item')
async def session_upsert_context_item(payload: dict):
    """Upsert a context item by item_id (idempotent).

    Expects JSON: {item_id, session_id, type, payload_json, created_at?}
    """
    item_id = payload.get('item_id')
    session_id = payload.get('session_id')
    type_ = payload.get('type')
    payload_json = payload.get('payload_json')
    created_at = payload.get('created_at')

    if not item_id or not session_id or type_ is None or payload_json is None:
        raise HTTPException(status_code=400, detail='Missing required fields')

    # Try ContextItemStore upsert if available
    try:
        from CognitiveRAG.session_memory.context_items import ContextItemStore
        cstore = ContextItemStore()
        try:
            cstore.upsert_item(item_id, session_id, type_, json.dumps(payload_json), created_at)
            # upsert_item is implemented as insert-or-update; we can't know created vs updated easily — return 'inserted' for now
            return {'status': 'inserted'}
        except Exception:
            # fallback to manual insert/update
            try:
                existing = cstore.get_item(item_id)
                if existing:
                    cstore.upsert_item(item_id, session_id, type_, json.dumps(payload_json), created_at)
                    return {'status': 'updated'}
                else:
                    cstore.upsert_item(item_id, session_id, type_, json.dumps(payload_json), created_at)
                    return {'status': 'inserted'}
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: write to data/session_memory/context_item_<item_id>.json
    workdir = os.path.join(os.getcwd(), 'data', 'session_memory')
    os.makedirs(workdir, exist_ok=True)
    raw_path = os.path.join(workdir, f'context_item_{item_id}.json')
    item = None
    if os.path.exists(raw_path):
        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                item = json.load(f)
        except Exception:
            item = None
    if item is None:
        item = {'item_id': item_id, 'session_id': session_id, 'type': type_, 'payload_json': payload_json, 'created_at': created_at}
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(item, f)
        return {'status': 'inserted'}
    else:
        item.update({'session_id': session_id, 'type': type_, 'payload_json': payload_json, 'created_at': created_at})
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(item, f)
        return {'status': 'updated'}

    try:
        patterns = promote_session_summaries(request.session_id, dry_run=False)
        ids = [p.pattern_id for p in patterns]
        return PromoteResponse(promoted_count=len(ids), promoted_pattern_ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {e}")
