import os
from bertopic import BERTopic
import networkx as nx
from pathlib import Path
import pickle

# Assumes document chunks are available from the ingestion/indexing step
# We'll load them from the vector store or from a saved file if available
CHUNKS_PATH = "./vector_chunks.pkl"
TOPIC_MODEL_PATH = "./bertopic_model"
GRAPH_PATH = "./knowledge_graph.gpickle"

# Helper to load chunks (list of text)
def load_chunks():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            return pickle.load(f)
    else:
        # Fallback: scan source_documents for .txt/.md/.json and chunk as in ingestion
        from CognitiveRAG.generated_files.knowledge_corpus_builder import ingest_files
        docs = ingest_files("./source_documents")
        texts = [text for text, meta in docs]
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(texts, f)
        return texts

def main():
    # --- Load Chunks ---
    print("Loading document chunks for topic modeling and graph...")
    docs = load_chunks()
    print(f"Loaded {len(docs)} chunks.")

    # --- BERTopic Topic Modeling ---
    print("Fitting BERTopic model...")
    topic_model = BERTopic(verbose=True)
    topics, _ = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()
    print(f"Discovered {len(topic_info)-1} topics (excluding outliers).")
    # Save topic model
    topic_model.save(TOPIC_MODEL_PATH)

    # --- Identify Emerging Topics ---
    emerging_topics = []
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue
        size = row['Count']
        if size < 5:
            emerging_topics.append(topic_id)
    print(f"Emerging topics (few documents): {emerging_topics}")

    # --- Build Knowledge Graph ---
    print("Building knowledge graph...")
    G = nx.Graph()
    # Add document nodes and topic nodes
    for idx, doc_id in enumerate(range(len(docs))):
        G.add_node(f"doc_{doc_id}", type='document', label=f"doc_{doc_id}")
    for t in set(topics):
        if t == -1:
            continue
        keywords = [kw for kw, _ in topic_model.get_topic(t)]
        label = f"Topic{t}: " + ", ".join(keywords[:3])
        G.add_node(f"topic_{t}", type='topic', label=label)
    # Connect docs to their topic
    for doc_id, t in enumerate(topics):
        if t == -1:
            continue
        G.add_edge(f"doc_{doc_id}", f"topic_{t}", relation="has_topic")
    # Add concept nodes (top keywords per topic)
    for t in set(topics):
        if t == -1:
            continue
        keywords = [kw for kw, _ in topic_model.get_topic(t)]
        for kw in keywords[:5]:
            concept_node = f"concept_{kw}"
            if not G.has_node(concept_node):
                G.add_node(concept_node, type='concept', label=kw)
            G.add_edge(concept_node, f"topic_{t}", relation="keyword_in_topic")
    # Save graph
    nx.write_gpickle(G, GRAPH_PATH)
    print(f"Knowledge graph saved to {GRAPH_PATH}.")

    # --- Analogical Reasoning Helper ---
    def find_related(concept):
        node = f"concept_{concept}"
        if not G.has_node(node):
            return []
        neighbors = list(G.neighbors(node))
        related_concepts = set()
        for n in neighbors:
            for m in G.neighbors(n):
                if m.startswith("concept_") and m != node:
                    related_concepts.add(G.nodes[m]['label'])
        return list(related_concepts)

    # Example usage
    test_concept = "quantum"
    rel = find_related(test_concept)
    print(f"Concepts related to '{test_concept}' via graph: {rel}")

if __name__ == "__main__":
    main()
