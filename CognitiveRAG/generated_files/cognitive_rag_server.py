import os
import requests
import uvicorn
import numpy as np
from typing import List, TypedDict
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# LangChain and LangGraph imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j.graphs import Neo4jGraph
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Analysis and ML imports
from bertopic import BERTopic
from neo4j import GraphDatabase
from sklearn.ensemble import IsolationForest

# --- 0. SETUP AND INITIALIZATION ---
print("--- Initializing Cognitive RAG Server ---")
load_dotenv()

# --- API MODELS ---
class IngestDocumentRequest(BaseModel):
    path: str

class IngestMcpRequest(BaseModel):
    url: str

class TransformPromptRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class TransformPromptResponse(BaseModel):
    enriched_knowledge: str
    status: str

# --- INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="Cognitive RAG System",
    description="An advanced agentic system for knowledge ingestion, reasoning, and discovery.",
    version="1.0.0",
)

# --- GLOBAL CONSTANTS AND MODELS ---
SOURCE_DIRECTORY = "./source_documents"
VECTOR_STORE_PATH = "./vector_store_chroma"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100

# Initialize models and databases once on startup
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
graph_db = Neo4jGraph()
vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
vector_retriever = vector_store.as_retriever()

# --- WORKFLOW 1: CORPUS INGESTION AND MANAGEMENT ---

def _process_and_store_chunks(chunks: list, source_id: str):
    """Helper function to process chunks and store them in Neo4j and ChromaDB."""
    print(f"Processing {len(chunks)} chunks from '{source_id}'...")
    if not chunks:
        return

    vector_store.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to ChromaDB.")

    try:
        llm_transformer = LLMGraphTransformer(llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini"))
        graph_documents = llm_transformer.convert_to_graph_documents(chunks)
        graph_db.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
        print(f"Added {len(graph_documents)} graph documents to Neo4j.")
    except Exception as e:
        print(f"Error adding to Neo4j: {e}")

def ingest_documents_task(path: str):
    """Background task to ingest local documents."""
    print(f"--- Starting Document Ingestion from path: {path} ---")
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at {path}.")
    
    if not os.listdir(path):
        with open(os.path.join(path, "placeholder.txt"), "w") as f:
            f.write("This is a placeholder. Add your text files here.")
        print("Created a placeholder document.")

    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True, use_multithreading=True)
    documents = loader.load()

    if not documents:
        print("No documents found to ingest.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    _process_and_store_chunks(chunks, source_id=path)
    print("--- Document Ingestion Complete ---")

def ingest_mcp_server_task(url: str):
    """Background task to ingest content from an MCP server."""
    print(f"--- Starting MCP Server Ingestion from URL: {url} ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(separator='\n', strip=True)
        
        if not content:
            print("No content fetched from MCP server.")
            return

        mcp_document = Document(page_content=content, metadata={"source": url})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents([mcp_document])
        _process_and_store_chunks(chunks, source_id=url)
        print("--- MCP Server Ingestion Complete ---")
    except Exception as e:
        print(f"An error occurred during MCP ingestion: {e}")

# --- WORKFLOW 2: INFERENCE AND DISCOVERY ---

class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    retrieved_context: List
    contradictions_found: bool
    novel_docs: List
    analogical_context: List
    final_context: str
    recursion_depth: int

def query_decomposer(state: AgentState):
    print("---NODE: Query Decomposer---")
    prompt = f"Decompose the user query into up to 3 distinct sub-queries. If simple, return the original query in a list. Query: \"{state['query']}\". Return a Python list of strings."
    response = llm.invoke(prompt)
    sub_queries = eval(response.content)
    return {"sub_queries": sub_queries, "recursion_depth": 1}

def hybrid_retriever(state: AgentState):
    print(f"---NODE: Hybrid Retriever (Depth: {state['recursion_depth']})---")
    all_docs =
    for q in state["sub_queries"]:
        all_docs.extend(vector_retriever.invoke(q))
        try:
            graph_data = graph_db.query(
                "CALL db.index.fulltext.queryNodes('chunk_fulltext', $q, {limit: 2}) YIELD node RETURN node.text AS text",
                params={"q": q}
            )
            for record in graph_data:
                all_docs.append(Document(page_content=record['text']))
        except Exception as e:
            print(f"Graph retrieval failed for query '{q}': {e}")
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    return {"retrieved_context": unique_docs}

def contradiction_checker(state: AgentState):
    print("---NODE: Contradiction Checker---")
    context_text = "\n---\n".join([doc.page_content for doc in state["retrieved_context"]])
    prompt = f"Analyze the following text for direct contradictions. Context: \"{context_text}\". Respond with only 'YES' or 'NO'."
    response = llm.invoke(prompt)
    return {"contradictions_found": "YES" in response.content}

def novelty_detector_node(state: AgentState):
    print("---NODE: Novelty Detector---")
    novelty_detector = IsolationForest(contamination='auto', random_state=42)
    all_embeddings = np.array(vector_store.get(include=['embeddings'])['embeddings'])
    
    if len(state["retrieved_context"]) == 0 or len(all_embeddings) == 0:
        return {"novel_docs":}
        
    novelty_detector.fit(all_embeddings)
    context_embeddings = embeddings.embed_documents([doc.page_content for doc in state["retrieved_context"]])
    scores = novelty_detector.predict(context_embeddings)
    
    novel_docs = [doc for doc, score in zip(state["retrieved_context"], scores) if score == -1]
    print(f"Found {len(novel_docs)} potentially novel documents.")
    return {"novel_docs": novel_docs}

def analogical_reasoner(state: AgentState):
    print("---NODE: Analogical Reasoner---")
    prompt = f"The user is asking about: '{state['query']}'. What is a structurally similar problem from a different domain? Generate one analogy and create 2 search queries based on it. Return a Python list of strings."
    response = llm.invoke(prompt)
    analogical_queries = eval(response.content)
    
    analogical_docs =
    for query in analogical_queries:
        analogical_docs.extend(vector_retriever.invoke(query))
    
    return {"analogical_context": analogical_docs}

def context_aggregator(state: AgentState):
    print("---NODE: Context Aggregator---")
    final_docs = state["retrieved_context"] + state.get("novel_docs",) + state.get("analogical_context",)
    unique_docs = list({doc.page_content: doc for doc in final_docs}.values())
    context_text = "\n---\n".join([doc.page_content for doc in unique_docs])
    
    if len(context_text) > 12000:
        print("Context is long, performing summarization...")
        summary = llm.invoke(f"Summarize the key points from the following text:\n\n{context_text}").content
        return {"final_context": summary}
    return {"final_context": context_text}

def saturation_controller(state: AgentState):
    print("---EDGE: Saturation Controller---")
    if state['recursion_depth'] >= 2:
        return "aggregate"
    if state['contradictions_found'] or not state['novel_docs']:
        return "analogize"
    return "aggregate"

# Build the inference graph
inference_workflow = StateGraph(AgentState)
inference_workflow.add_node("decompose_query", query_decomposer)
inference_workflow.add_node("retrieve_context", hybrid_retriever)
inference_workflow.add_node("check_contradictions", contradiction_checker)
inference_workflow.add_node("detect_novelty", novelty_detector_node)
inference_workflow.add_node("analogize", analogical_reasoner)
inference_workflow.add_node("aggregate_context", context_aggregator)
inference_workflow.set_entry_point("decompose_query")
inference_workflow.add_edge("decompose_query", "retrieve_context")
inference_workflow.add_edge("retrieve_context", "check_contradictions")
inference_workflow.add_edge("check_contradictions", "detect_novelty")
inference_workflow.add_edge("analogize", "aggregate_context")
inference_workflow.add_conditional_edges(
    "detect_novelty",
    saturation_controller,
    {"aggregate": "aggregate_context", "analogize": "analogize"}
)
inference_workflow.add_edge("aggregate_context", END)
app_inference = inference_workflow.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))

# --- API ENDPOINTS ---

@app.post("/ingest-document/", status_code=202)
async def ingest_document_endpoint(request: IngestDocumentRequest, background_tasks: BackgroundTasks):
    """API endpoint to ingest documents from a local path."""
    background_tasks.add_task(ingest_documents_task, request.path)
    return {"message": f"Document ingestion started for path: {request.path}. This will be processed in the background."}

@app.post("/ingest-mcp/", status_code=202)
async def ingest_mcp_endpoint(request: IngestMcpRequest, background_tasks: BackgroundTasks):
    """API endpoint to ingest content from an MCP server URL."""
    background_tasks.add_task(ingest_mcp_server_task, request.url)
    return {"message": f"MCP server ingestion started for URL: {request.url}. This will be processed in the background."}

@app.post("/transform-prompt/", response_model=TransformPromptResponse)
async def transform_prompt_endpoint(request: TransformPromptRequest):
    """API endpoint to process a user query and return enriched knowledge."""
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        final_state = app_inference.invoke({"query": request.query}, config=config)
        enriched_knowledge = final_state.get("final_context", "No context could be generated.")
        return {"enriched_knowledge": enriched_knowledge, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during inference: {str(e)}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # This block allows you to run the server directly
    print("Starting FastAPI server...")
    # Create a full-text index on startup if it doesn't exist
    try:
        graph_db.query("CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]")
        print("Neo4j full-text index ensured.")
    except Exception as e:
        print(f"Could not create Neo4j index. Please ensure DB is running. Error: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8001)
