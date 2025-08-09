# CognitiveRAG

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An advanced agentic Retrieval-Augmented Generation (RAG) system for knowledge discovery**

CognitiveRAG combines multi-agent orchestration, hybrid retrieval, and explainable synthesis to answer complex queries using both internal and external knowledge sources. The system is designed to discover "unknown unknowns" through sophisticated pattern detection and cross-domain analogical reasoning.

## ?? Key Features

- **?? Multi-Agent Architecture**: Specialized agents for planning, synthesis, and reflection
- **?? Hybrid Retrieval**: Combines BM25, vector search, and web search for comprehensive coverage
- **?? Knowledge Graph**: Builds topic-based knowledge graphs from ingested documents
- **?? Explainable Results**: Provides complete source attribution for all answers
- **? Pluggable LLMs**: Support for both Ollama (local) and OpenAI models
- **?? RESTful API**: FastAPI with automatic interactive documentation
- **?? Real-time Processing**: Live document updates and streaming responses

## ??? System Architecture
???????????????????    ???????????????????    ???????????????????
?   FastAPI       ?    ?   Multi-Agent   ?    ?   Knowledge     ?
?   Server        ??????   Orchestrator  ??????   Base          ?
???????????????????    ???????????????????    ???????????????????
                              ?                         ?
                              ?                         ?
                    ???????????????????    ???????????????????
                    ?   Hybrid        ?    ?   Vector Store  ?
                    ?   Retriever     ?    ?   + Graph DB    ?
                    ???????????????????    ???????????????????
### Core Components

- **FastAPI Server**: REST endpoints for document ingestion and querying
- **KnowledgeBase**: Document storage with BM25 index, vector store, topic modeling, and knowledge graph
- **Multi-Agent System**: Coordinated agents for planning, synthesis, and reflection
- **Hybrid Retriever**: Combines keyword, semantic, and web search strategies
- **LLM Provider**: Abstraction layer supporting multiple LLM backends

### Multi-Agent Workflow
User Query
    ?
??????????????????? ? Breaks query into logical steps
? PlannerAgent    ?
???????????????????
    ?
??????????????????? ? BM25 + Vector + Web search
? HybridRetriever ?
???????????????????
    ?
??????????????????? ? Generates evidence-based answer
? SynthesisAgent  ?
???????????????????
    ?
??????????????????? ? Quality assurance feedback loop
? ReflectionAgent ?
???????????????????
    ?
Final Answer with Sources
## ?? Prerequisites

- **Python 3.11+**
- **Git**
- **Ollama** (for local LLM inference) OR **OpenAI API Key**

## ??? Installation

### 1. Clone the Repository
git clone https://github.com/yourusername/CognitiveRAG.git
cd CognitiveRAG
### 2. Install Dependencies
pip install -r CognitiveRAG/requirements.txt
### 3. Configure LLM Provider

#### Option A: Using Ollama (Recommended for Local Development)

1. Install [Ollama](https://ollama.ai/)
2. Pull required models:ollama pull llama3
ollama pull gemma:2b
ollama pull nomic-embed-text3. Edit `CognitiveRAG/config.py`:LLM_PROVIDER = "ollama"
PLANNER_MODEL = "llama3"
SYNTHESIS_MODEL = "llama3"
REFLECTION_MODEL = "gemma:2b"
EMBEDDING_MODEL = "nomic-embed-text"
#### Option B: Using OpenAI

1. Set your API key as an environment variable:export OPENAI_API_KEY="your-api-key-here"2. Edit `CognitiveRAG/config.py`:LLM_PROVIDER = "openai"
PLANNER_MODEL = "gpt-4o-mini"
SYNTHESIS_MODEL = "gpt-4o-mini"
REFLECTION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
### 4. Start the Server
python -m uvicorn CognitiveRAG.main_server:app --reload --host 127.0.0.1 --port 8080
### 5. Verify Installation
curl http://127.0.0.1:8080/health
Expected response:{"status": "ok"}
## ?? API Documentation

### Interactive Documentation

Visit `http://127.0.0.1:8080/docs` when the server is running for full interactive API documentation.

### Endpoints

#### `POST /ingest` - Document Ingestion

Upload documents to build the knowledge base.

**Request:**curl -F "files=@document1.txt" -F "files=@document2.pdf" \
  http://127.0.0.1:8080/ingest
**Response:**
{
  "status": "success",
  "message": "Successfully ingested and rebuilt knowledge base with 2 files.",
  "files_processed": ["document1.txt", "document2.pdf"]
}
**Supported formats:** `.txt`, `.md`, `.json`

#### `POST /query` - Query Processing

Submit queries to the RAG system.

**Request:**curl -X POST http://127.0.0.1:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings about climate change?"}'
**Response:**
{
  "answer": "Based on the provided documents, the main findings about climate change include...",
  "sources": ["climate_report.pdf", "research_paper.txt"]
}
#### `GET /health` - Health Check
curl http://127.0.0.1:8080/health
**Response:**
{"status": "ok"}
#### `GET /` - System Information
curl http://127.0.0.1:8080/
**Response:**
{
  "message": "Cognitive RAG System",
  "status": "running",
  "version": "1.0.0"
}
## ?? Docker Deployment

### Build Docker Image
docker build -t cognitive-rag .
### Run Container
# For Ollama (requires host network access)
docker run -p 8000:8000 --network host cognitive-rag

# For OpenAI
docker run -p 8000:8000 -e OPENAI_API_KEY="your-key" cognitive-rag
## ?? Project Structure
CognitiveRAG/
??? __init__.py              # Package initialization
??? main_server.py           # FastAPI server and endpoints
??? main.py                  # Development server launcher
??? config.py                # Configuration settings
??? knowledge_base.py        # Document storage and indexing
??? agents.py                # Multi-agent orchestration
??? retriever.py             # Hybrid retrieval system
??? llm_provider.py          # LLM abstraction layer
??? utils.py                 # Utility functions
??? requirements.txt         # Python dependencies
??? Dockerfile               # Container configuration
??? generated_files/         # Additional implementation files
    ??? cognitive_rag_server.py
    ??? rag_qa.py
    ??? evaluation_suite.py
    ??? ...
## ?? Configuration

The system is configured through `CognitiveRAG/config.py`. Key settings include:

### LLM Configuration# Provider selection
LLM_PROVIDER = "ollama"  # or "openai"

# Model assignments
PLANNER_MODEL = "llama3"
SYNTHESIS_MODEL = "llama3"
REFLECTION_MODEL = "gemma:2b"
EMBEDDING_MODEL = "nomic-embed-text"
### Retrieval Settings# Document chunking
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100

# Retrieval limits
BM25_TOP_K = 5
VECTOR_TOP_K = 5

# Agent behavior
MAX_RECURSION_DEPTH = 3
### Storage PathsCORPUS_DIR = "./corpus_data"
SOURCE_DOCUMENTS_DIR = "./corpus_data/source_documents"
INDEX_DIR = "./corpus_data/indexes"
## ?? Development

### Running for Development

**Visual Studio Code:**
1. Set `CognitiveRAG/main.py` as the startup file
2. Press F5 to start debugging

**Command Line:**# Development server with auto-reload
python -m uvicorn CognitiveRAG.main_server:app --reload --port 8080

# With debug logging
python -m uvicorn CognitiveRAG.main_server:app --log-level debug --port 8080
### Testing the System

1. **Start the server** following installation instructions
2. **Upload test documents** via `/ingest` endpoint
3. **Submit queries** via `/query` endpoint
4. **Monitor responses** and source attribution

## ?? Data Flow

### Document IngestionUpload Files ? /ingest endpoint ? KnowledgeBase.build() ? 
BM25 Index + Vector Store + Topic Model + Knowledge Graph
### Query ProcessingUser Query ? PlannerAgent ? HybridRetriever ? SynthesisAgent ? 
ReflectionAgent ? Final Answer with Sources
## ?? Troubleshooting

### Common Issues

**Import Errors:**
- Ensure you're running from the parent directory of `CognitiveRAG`
- Check for file naming conflicts (avoid files named `CognitiveRAG.py`)

**Server Won't Start:**
- Verify Python 3.11+ is installed: `python --version`
- Check port availability: `netstat -an | grep 8080`
- Install dependencies: `pip install -r CognitiveRAG/requirements.txt`

**LLM Connection Issues:**
- **Ollama**: Ensure Ollama is running (`ollama serve`) and models are pulled
- **OpenAI**: Verify API key is set and valid

**Empty Knowledge Base:**
- Upload documents using the `/ingest` endpoint before querying
- Check that documents are in supported formats (`.txt`, `.md`, `.json`)
- Verify the `source_documents` directory contains files

**Memory Issues:**
- Reduce `CHUNK_SIZE` in configuration for large documents
- Lower `BM25_TOP_K` and `VECTOR_TOP_K` values
- Use smaller LLM models (e.g., `gemma:2b` instead of `llama3`)

## ?? Advanced Features

### Knowledge Discovery Pipeline

The system implements a sophisticated discovery pipeline for finding "unknown unknowns":

1. **External-First Discovery**: Template-based exploration
2. **Mechanistic Pattern Detection**: Statistical signal identification
3. **Cross-Domain Analogical Retrieval**: Structural similarity matching
4. **Evidence-Grounded Reflection**: LLM reasoning anchored in evidence
5. **Recursive Refinement**: Iterative deepening with saturation controls

### Performance Characteristics

- **Query Throughput**: Optimized for real-time responses
- **Scalability**: Supports large document corpora
- **Discovery Effectiveness**: High precision for complex research queries
- **Source Attribution**: Complete traceability for all claims

## ?? Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m "Add feature"`
5. **Push to the branch**: `git push origin feature-name`
6. **Submit a pull request**

### Development Guidelines

- Follow Python PEP 8 style guidelines
- Add comprehensive docstrings to new functions
- Include unit tests for new features
- Update documentation for API changes

## ?? License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ?? Resources

- **[API Documentation](http://127.0.0.1:8080/docs)** (when server is running)
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)**
- **[LangChain Documentation](https://python.langchain.com/)**
- **[Ollama Models](https://ollama.ai/library)**

## ?? Important Notes

- This is a research prototype designed for exploration and development
- For production deployment, implement proper authentication and security measures
- Monitor resource usage when processing large document corpora
- Review and adjust LLM model choices based on your hardware capabilities

---

**Built with ?? for the AI research community**
