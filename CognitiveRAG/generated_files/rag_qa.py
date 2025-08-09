from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
from CognitiveRAG.generated_files.knowledge_corpus_builder import ingest_files
from CognitiveRAG.generated_files.topic_graph import main as topic_graph_main
import os

LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Placeholder, replace with local model

# Load LLM
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
except Exception as e:
    tokenizer = None
    model = None
    print(f"[WARNING] Could not load LLM: {e}")

def generate_text(prompt, max_tokens=256):
    if not model or not tokenizer:
        return "[LLM not available]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

class RAGAgent:
    def __init__(self, retriever, knowledge_graph=None):
        self.retriever = retriever
        self.KG = knowledge_graph
        self.intermediate_answers = {}
    def plan_query(self, query):
        prompt = ("You are a planner agent. You will break a complex question into steps.\n"
                  f"User question: {query}\n"
                  "Plan steps (concise list):\n"
                  "1.")
        plan_text = generate_text(prompt, max_tokens=100)
        steps = []
        for line in plan_text.splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                step_desc = line.lstrip("0123456789.-) ").strip()
                if step_desc:
                    steps.append(step_desc)
        if not steps:
            steps = [query]
        return steps
    def answer(self, query):
        final_answer = ""
        plan = self.plan_query(query)
        context_so_far = ""
        for i, step in enumerate(plan, start=1):
            subquery = step
            if "<" in subquery and ">" in subquery:
                subquery = subquery.replace("<", "").replace(">", "")
            hits = self.retriever.retrieve(subquery, top_k=5)
            docs_text = [self.retriever.docs[doc_id] for doc_id, _, _ in hits]
            if not docs_text:
                docs_text = []  # Placeholder for external search
            evidence = ""
            for text in docs_text[:3]:
                snippet = text[:500].replace('\n',' ')
                evidence += snippet + "\n"
            if not evidence:
                evidence = "(No relevant information found.)"
            prompt = (f"Question: {subquery}\n"
                      f"Context:\n{evidence}\n"
                      "Answer:")
            subanswer = generate_text(prompt, max_tokens=200)
            self.intermediate_answers[i] = {"question": subquery, "answer": subanswer}
            context_so_far += f"Step {i}: {subanswer}\n"
        if len(plan) > 1:
            combined_prompt = ("Combine the following step answers to address the original question:\n"
                                f"Question: {query}\n")
            for i, ans in self.intermediate_answers.items():
                combined_prompt += f"Step {i} answer: {ans['answer']}\n"
            combined_prompt += "Final answer (with any necessary qualifications):"
            final_answer = generate_text(combined_prompt, max_tokens=150)
        else:
            final_answer = self.intermediate_answers.get(1, {}).get("answer", "")
        if "No relevant information" in final_answer or final_answer.strip() == "":
            final_answer = ("I'm sorry, but I cannot find sufficient information to answer that. "
                            "It might be an open question or beyond the current knowledge base.")
        return final_answer

# HybridRetriever stub for integration
def get_retriever():
    # Load BM25 and chunks
    BM25_PATH = "./bm25_index.pkl"
    CHUNKS_PATH = "./vector_chunks.pkl"
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
    # Build doc_id mapping
    doc_ids = [f"doc{i}" for i in range(len(chunks))]
    docs = {doc_id: chunk for doc_id, chunk in zip(doc_ids, chunks)}
    class HybridRetriever:
        def __init__(self, docs, bm25):
            self.docs = docs
            self.bm25 = bm25
            self.doc_ids = list(docs.keys())
        def retrieve(self, query, top_k=5):
            from nltk.tokenize import word_tokenize
            q_tokens = [w.lower() for w in word_tokenize(query)]
            bm25_scores = self.bm25.get_scores(q_tokens)
            bm25_hits = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
            results = []
            for i in bm25_hits:
                doc_id = self.doc_ids[i]
                score = bm25_scores[i]
                results.append((doc_id, float(score), "bm25"))
            return results
    return HybridRetriever(docs, bm25)

# Load knowledge graph if available
def get_knowledge_graph():
    import networkx as nx
    GRAPH_PATH = "./knowledge_graph.gpickle"
    if os.path.exists(GRAPH_PATH):
        return nx.read_gpickle(GRAPH_PATH)
    return None

# Instantiate agent
def main():
    retriever = get_retriever()
    KG = get_knowledge_graph()
    agent = RAGAgent(retriever, knowledge_graph=KG)
    query = input("Enter your question: ")
    answer = agent.answer(query)
    print("\n=== Answer ===")
    print(answer)

if __name__ == "__main__":
    main()
