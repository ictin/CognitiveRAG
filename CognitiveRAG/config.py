# CognitiveRAG/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment and API Keys ---
# OpenAI key is now optional, used only if specified as the provider
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM and Embedding Models ---
# Provider can be 'ollama' or 'openai'
LLM_PROVIDER = "ollama"
OLLAMA_BASE_URL = "http://localhost:11434"

# Configure models for each purpose
# For Ollama, specify the model name (e.g., "llama3", "gemma:2b")
# For OpenAI, specify the model name (e.g., "gpt-4o-mini")
PLANNER_MODEL = "llama3"
SYNTHESIS_MODEL = "llama3"
REFLECTION_MODEL = "gemma:2b"
EMBEDDING_MODEL = "nomic-embed-text"  # Recommended Ollama embedding model

# --- Directories and Paths ---
CORPUS_DIR = "./corpus_data"
SOURCE_DOCUMENTS_DIR = os.path.join(CORPUS_DIR, "source_documents")
INDEX_DIR = os.path.join(CORPUS_DIR, "indexes")
VECTOR_STORE_PATH = os.path.join(INDEX_DIR, "chroma_vector_store")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
DOC_STORE_PATH = os.path.join(INDEX_DIR, "doc_store.pkl")
TOPIC_MODEL_PATH = os.path.join(INDEX_DIR, "bertopic_model")
KNOWLEDGE_GRAPH_PATH = os.path.join(INDEX_DIR, "knowledge_graph.gpickle")

# --- Ingestion and Chunking ---
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100

# --- Retrieval Settings ---
BM25_TOP_K = 5
VECTOR_TOP_K = 5

# --- Multi-Agent Orchestration ---
MAX_RECURSION_DEPTH = 3

# --- Signal Detection ---
EMERGING_TOPIC_THRESHOLD = 5

# --- Web Search (MCP) ---
WEB_SEARCH_ENABLED = True
