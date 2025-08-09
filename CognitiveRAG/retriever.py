# CognitiveRAG/retriever.py
from typing import List
from langchain_core.documents import Document
from duckduckgo_search import DDGS
from . import config
from . import utils
from .knowledge_base import kb

class HybridRetriever:
    """
    Performs hybrid retrieval from BM25, vector store, and external web search.
    """
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Combines results from BM25, vector search, and web search.
        """
        all_docs = []
        
        # 1. BM25 Retrieval (Keyword)
        bm25_results = []
        try:
            if self.kb and self.kb.bm25_index and self.kb.doc_store:
                tokenized_query = utils.tokenize(query)
                doc_scores = self.kb.bm25_index.get_scores(tokenized_query)
                
                # Get the doc_ids and scores
                doc_id_list = list(self.kb.doc_store.keys())
                scored_docs = sorted(zip(doc_scores, doc_id_list), reverse=True)
                
                top_doc_ids = [doc_id for score, doc_id in scored_docs[:top_k] if score > 0]
                bm25_results = [self.kb.doc_store[doc_id] for doc_id in top_doc_ids]
        except Exception as e:
            print(f"Warning: BM25 retrieval failed: {e}")
        
        # 2. Vector Retrieval (Semantic)
        vector_results = []
        try:
            if self.kb and self.kb.vector_store:
                vector_results = self.kb.vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"Warning: Vector retrieval failed: {e}")

        # 3. Web Retrieval (External)
        web_results = []
        try:
            if config.WEB_SEARCH_ENABLED:
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(query, max_results=top_k))
                    web_results = [
                        Document(page_content=r['body'], metadata={'source': r['href'], 'title': r['title']})
                        for r in search_results
                    ]
        except Exception as e:
            print(f"Warning: Web search failed: {e}")

        # 4. Combine and de-duplicate results
        all_docs = bm25_results + vector_results + web_results
        try:
            unique_docs = {doc.page_content: doc for doc in all_docs}.values()
            unique_docs_list = list(unique_docs)
        except Exception as e:
            print(f"Warning: Could not deduplicate documents: {e}")
            unique_docs_list = all_docs
        
        print(f"Retrieved {len(unique_docs_list)} unique documents ({len(bm25_results)} BM25, {len(vector_results)} vector, {len(web_results)} web).")
        return unique_docs_list

# Singleton instance - only create if kb is available
try:
    if kb is not None:
        retriever = HybridRetriever(kb)
    else:
        retriever = None
except Exception as e:
    print(f"Warning: Could not initialize retriever: {e}")
    retriever = None
