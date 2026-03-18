# CognitiveRAG/main_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from .knowledge_base import kb
import os
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


@app.post('/promote_session', response_model=PromoteResponse)
async def promote_session(request: PromoteRequest):
    """Operator endpoint: promote session summaries into durable reasoning memory.

    This is deliberately explicit and additive. It does not remove or modify raw session memory.
    """
    try:
        from CognitiveRAG.session_memory.promotion_bridge import promote_session_summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion bridge not available: {e}")


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

    try:
        patterns = promote_session_summaries(request.session_id, dry_run=False)
        ids = [p.pattern_id for p in patterns]
        return PromoteResponse(promoted_count=len(ids), promoted_pattern_ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {e}")
