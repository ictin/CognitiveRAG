# CognitiveRAG/main_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from .knowledge_base import kb
from .agents import Orchestrator
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
        orchestrator = Orchestrator()
        final_state = orchestrator.run(request.query)
        sources = sorted(list({doc.metadata.get('source', 'N/A') for doc in final_state['context']}))
        return QueryResponse(
            answer=final_state['answer'],
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
