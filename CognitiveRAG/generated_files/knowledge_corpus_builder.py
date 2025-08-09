import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j.graphs import Neo4jGraph
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

# Load environment variables
load_dotenv()

# --- Constants ---
SOURCE_DIRECTORY = "./source_documents"
VECTOR_STORE_PATH = "./vector_store_chroma"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100

# --- Document Ingestion and Chunking ---
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_files(folder):
    """Read all .txt, .md, .json files in folder and return list of (text, meta)"""
    docs = []
    for filepath in Path(folder).glob("*"):
        if filepath.suffix.lower() in [".txt", ".md"]:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            if filepath.suffix.lower() == ".md":
                content = "\n".join(
                    line for line in content.splitlines() 
                    if not line.strip().startswith("```")
                )
            for chunk in chunk_text(content):
                meta = {"source_file": filepath.name}
                docs.append((chunk, meta))
        elif filepath.suffix.lower() == ".json":
            try:
                data = json.loads(filepath.read_text(encoding='utf-8'))
            except Exception:
                continue
            def find_text(obj):
                texts = []
                if isinstance(obj, str):
                    if len(obj) > 100:
                        texts.append(obj)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        texts.extend(find_text(v))
                elif isinstance(obj, list):
                    for v in obj:
                        texts.extend(find_text(v))
                return texts
            texts = find_text(data)
            for text in texts:
                for chunk in chunk_text(text):
                    meta = {"source_file": filepath.name}
                    docs.append((chunk, meta))
    return docs

def main():
    # --- Ingest Documents ---
    print("Loading documents (.txt, .md, .json)...")
    docs = ingest_files(SOURCE_DIRECTORY)
    if not docs:
        print(f"No documents found in {SOURCE_DIRECTORY}. Please add files to proceed.")
        if not os.path.exists(SOURCE_DIRECTORY):
            os.makedirs(SOURCE_DIRECTORY)
        with open(os.path.join(SOURCE_DIRECTORY, "placeholder.txt"), "w") as f:
            f.write("This is a placeholder document for the Cognitive RAG system.")
        print("Created a placeholder document. Please add your actual documents and re-run.")
        docs = ingest_files(SOURCE_DIRECTORY)
    print(f"Loaded {len(docs)} chunks from files.")

    # --- Prepare for BM25 ---
    print("Building BM25 index...")
    corpus_tokens = [word_tokenize(text.lower()) for text, _ in docs]
    bm25 = BM25Okapi(corpus_tokens)

    # --- Prepare for Vector Store ---
    print("Building vector store with ChromaDB...")
    # Convert to LangChain Document objects for Chroma
    from langchain_core.documents import Document
    lc_docs = [Document(page_content=text, metadata=meta) for text, meta in docs]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    print(f"Vector store created and persisted at {VECTOR_STORE_PATH}.")

    # --- Build and Populate Knowledge Graph ---
    print("Connecting to Neo4j and building knowledge graph...")
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        graph = Neo4jGraph()
        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = llm_transformer.convert_to_graph_documents(lc_docs)
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        graph.query("CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]")
        print("Knowledge graph populated and indexed successfully.")
    except Exception as e:
        print(f"Error connecting to or populating Neo4j: {e}")
        print("Please ensure your Neo4j Docker container is running and credentials in.env are correct.")
        return

    print("\n--- Knowledge Corpus Build Complete ---")

if __name__ == "__main__":
    main()
