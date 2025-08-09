import os
import networkx as nx
from rank_bm25 import BM25Okapi
from bertopic import BERTopic
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from . import config
from . import utils
from .llm_provider import get_embeddings

class KnowledgeBase:
    """
    Manages all persistent data stores: Document Store, BM25 Index,
    Vector Store, Topic Model, and Knowledge Graph.
    """
    def __init__(self):
        self.doc_store: dict = {}
        self.bm25_index: BM25Okapi = None
        self.vector_store: Chroma = None
        self.topic_model: BERTopic = None
        self.knowledge_graph: nx.Graph = None
        try:
            self._load_all()
        except Exception as e:
            print(f"Warning: Could not load knowledge base components: {e}")

    def _load_all(self):
        """Load all components from disk if they exist."""
        print("--- Loading Knowledge Base ---")
        self.doc_store = utils.load_pickle(config.DOC_STORE_PATH) or {}
        self.bm25_index = utils.load_pickle(config.BM25_INDEX_PATH)
        self.topic_model = BERTopic.load(config.TOPIC_MODEL_PATH) if os.path.exists(config.TOPIC_MODEL_PATH) else None
        self.knowledge_graph = utils.load_pickle(config.KNOWLEDGE_GRAPH_PATH)

        # Initialize vector store with the configured embedding provider
        try:
            embeddings = get_embeddings()
            self.vector_store = Chroma(
                persist_directory=config.VECTOR_STORE_PATH,
                embedding_function=embeddings
            )
            print(f"Loaded {len(self.doc_store)} documents.")
            print(f"Vector store contains {self.vector_store._collection.count()} embeddings.")
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            self.vector_store = None

    def build(self):
        """Builds all indexes from the source documents directory."""
        print("--- Building Knowledge Base from Source ---")
        # 1. Ingest and chunk documents
        if not os.path.exists(config.SOURCE_DOCUMENTS_DIR):
            os.makedirs(config.SOURCE_DOCUMENTS_DIR)
            with open(os.path.join(config.SOURCE_DOCUMENTS_DIR, "placeholder.txt"), "w") as f:
                f.write("Add your documents here.")
            print("Created placeholder document.")

        documents = utils.ingest_files_from_folder(
            config.SOURCE_DOCUMENTS_DIR,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP
        )
        if not documents:
            print("No documents found to build knowledge base.")
            return

        # Create a unique ID for each document chunk
        for i, doc in enumerate(documents):
            doc.metadata["doc_id"] = f"doc_{i}"
        
        self.doc_store = {doc.metadata["doc_id"]: doc for doc in documents}
        utils.save_pickle(self.doc_store, config.DOC_STORE_PATH)
        print(f"Ingested and stored {len(self.doc_store)} document chunks.")

        # 2. Build BM25 index
        print("Building BM25 index...")
        tokenized_corpus = [utils.tokenize(doc.page_content) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        utils.save_pickle(self.bm25_index, config.BM25_INDEX_PATH)

        # 3. Build Vector Store (Chroma)
        print("Building vector store...")
        embeddings = get_embeddings()
        # Clear old collection if it exists, to rebuild
        if os.path.exists(config.VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(config.VECTOR_STORE_PATH)
        self.vector_store = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=config.VECTOR_STORE_PATH
        )
        print(f"Vector store built with {self.vector_store._collection.count()} embeddings.")

        # 4. Build Topic Model
        print("Building topic model...")
        self.topic_model = BERTopic(language="english", verbose=True)
        doc_contents = [doc.page_content for doc in documents]
        topics, _ = self.topic_model.fit_transform(doc_contents)
        self.topic_model.save(config.TOPIC_MODEL_PATH)
        print(f"Topic model built. Found {len(self.topic_model.get_topic_info())-1} topics.")

        # 5. Build Knowledge Graph
        print("Building knowledge graph...")
        self.knowledge_graph = nx.Graph()
        # This is a simplified KG. A real implementation would use LLM-based entity/relation extraction.
        for i, doc in enumerate(documents):
            doc_id = doc.metadata["doc_id"]
            topic_id = topics[i]
            self.knowledge_graph.add_node(doc_id, type='document', content=doc.page_content[:100])
            if topic_id != -1:
                topic_name = f"topic_{topic_id}"
                if not self.knowledge_graph.has_node(topic_name):
                    self.knowledge_graph.add_node(topic_name, type='topic', label=self.topic_model.get_topic(topic_id)[0][0])
                self.knowledge_graph.add_edge(doc_id, topic_name)
        utils.save_pickle(self.knowledge_graph, config.KNOWLEDGE_GRAPH_PATH)
        print("Knowledge graph built.")
        print("--- Knowledge Base Build Complete ---")

# Singleton instance - only create if being imported as module
try:
    kb = KnowledgeBase()
except Exception as e:
    print(f"Warning: Could not initialize knowledge base: {e}")
    kb = None
