import argparse
import math
import pickle
from CognitiveRAG.generated_files.knowledge_corpus_builder import ingest_files
from CognitiveRAG.generated_files.topic_graph import main as topic_graph_main
from CognitiveRAG.generated_files.rag_qa import RAGAgent, get_retriever, get_knowledge_graph
import os

# Load BM25 and vector index if available, else build
BM25_PATH = "./bm25_index.pkl"
CHUNKS_PATH = "./vector_chunks.pkl"

# --- Signal Detectors ---
def detect_rare_terms(query, bm25):
    tokens = [w.lower() for w in query.split()]
    rare_tokens = [t for t in tokens if t not in bm25.idf or bm25.idf[t] < math.log(2)]
    return rare_tokens

def main():
    parser = argparse.ArgumentParser(description="Unknown-Unknowns RAG QA System")
    parser.add_argument("question", type=str, help="User question to ask")
    parser.add_argument("--no-web", action="store_true", help="Disable external web search")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate steps and sources")
    args = parser.parse_args()

    # --- Load or Build Indexes ---
    if os.path.exists(BM25_PATH) and os.path.exists(CHUNKS_PATH):
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
    else:
        docs = ingest_files("./source_documents")
        chunks = [text for text, meta in docs]
        from rank_bm25 import BM25Okapi
        from nltk.tokenize import word_tokenize
        corpus_tokens = [word_tokenize(text.lower()) for text in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)

    # --- Signal Detection ---
    if args.verbose:
        print(f"[Query] {args.question}")
        rare_tokens = detect_rare_terms(args.question, bm25)
        if rare_tokens:
            print(f"[Signal] Rare terms detected in query: {rare_tokens}")
        print("----- Starting multi-agent reasoning -----")

    # --- Multi-Agent RAG Pipeline ---
    retriever = get_retriever()
    KG = get_knowledge_graph()
    agent = RAGAgent(retriever, knowledge_graph=KG)
    answer = agent.answer(args.question)
    print("\n=== Answer ===")
    print(answer)

if __name__ == "__main__":
    main()
