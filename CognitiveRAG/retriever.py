# CognitiveRAG/retriever.py
from typing import List
from langchain_core.documents import Document
from duckduckgo_search import DDGS
from . import config
from . import utils
from .knowledge_base import kb
from .retrieval_contract import RetrievedChunk

from .retrieval_modes import RetrievalMode, allowed_sources_for_mode

# Optional injected store instances (set by lifecycle when available)
_injected_task_store = None
_injected_profile_store = None
_injected_reasoning_store = None
_injected_episodic_store = None
_injected_web_promoted_store = None

def set_memory_stores(task_store=None, profile_store=None, reasoning_store=None, episodic_store=None, web_promoted_store=None):
    """Set injected memory store instances for retriever to reuse (optional).
    Small helper allowing lifecycle to provide shared store instances.
    """
    global _injected_task_store, _injected_profile_store, _injected_reasoning_store, _injected_episodic_store, _injected_web_promoted_store
    _injected_task_store = task_store
    _injected_profile_store = profile_store
    _injected_reasoning_store = reasoning_store
    _injected_episodic_store = episodic_store
    _injected_web_promoted_store = web_promoted_store


class HybridRetriever:
    """
    Performs hybrid retrieval from BM25, vector store, and external web search.
    """
    def __init__(self, knowledge_base, task_store=None, profile_store=None, reasoning_store=None, episodic_store=None):
        self.kb = knowledge_base
        # allow optional direct injection on instance creation
        self._task_store = task_store
        self._profile_store = profile_store
        self._reasoning_store = reasoning_store
        self._episodic_store = episodic_store

    def retrieve(self, query: str, top_k: int = 5, mode: RetrievalMode = RetrievalMode.FULL_MEMORY) -> List[Document]:
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

        # 3. Web Retrieval (External) — stage as web_evidence
        web_results = []
        try:
            if config.WEB_SEARCH_ENABLED:
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(query, max_results=top_k))
                    web_results = [
                        Document(page_content=r['body'], metadata={'source': r['href'], 'title': r['title'], 'source_type': 'web_evidence', 'document_kind': 'web_capture'})
                        for r in search_results
                    ]
        except Exception as e:
            print(f"Warning: Web search failed: {e}")

        # Apply mode-based filtering for immediate sources and also optionally include memory stores
        try:
            policy = allowed_sources_for_mode(mode)
            if not policy.get('bm25', True):
                bm25_results = []
            if not policy.get('vector', True):
                vector_results = []
            if not policy.get('web', True):
                web_results = []

            # If policy allows task/profile/reasoning, query those stores from local data files
            extra_memory_results = []
            if policy.get('task_profile_reasoning', False):
                try:
                    from CognitiveRAG.memory.task_store import TaskStore
                    from CognitiveRAG.memory.profile_store import ProfileStore
                    from CognitiveRAG.memory.reasoning_store import ReasoningStore
                    # Prefer injected instance, then instance-level injection, then canonical settings-based construction
                    from CognitiveRAG.core.settings import settings
                    task_store = _injected_task_store or self._task_store or TaskStore(settings.store.task_db_path)
                    profile_store = _injected_profile_store or self._profile_store or ProfileStore(settings.store.profile_db_path)
                    reasoning_store = _injected_reasoning_store or self._reasoning_store or ReasoningStore(settings.store.reasoning_db_path)

                    tresults = task_store.query(query, top_k=top_k)
                    presults = profile_store.query(query, top_k=top_k)
                    rresults = reasoning_store.query(query, top_k=top_k)

                    # convert dict results to Document-like objects
                    for r in tresults + presults + rresults:
                        try:
                            # some stores return dicts with 'text' and 'metadata'
                            page = r.get('text') or r.get('document') or ''
                            meta = r.get('metadata', {})
                            meta['source_type'] = r.get('source_type', meta.get('source_type', 'task'))
                            extra_memory_results.append(Document(page_content=page, metadata=meta))
                        except Exception:
                            continue
                except Exception as e:
                    print(f"Warning: could not query task/profile/reasoning stores: {e}")

            # If policy allows episodic, query episodic store
            if policy.get('episodic', False):
                try:
                    from CognitiveRAG.memory.episodic_store import EpisodicStore
                    from CognitiveRAG.core.settings import settings
                    episodic_store = _injected_episodic_store or self._episodic_store or EpisodicStore(settings.store.episodic_db_path)
                    eresults = episodic_store.query(query, top_k=top_k)
                    # EpisodicStore.query returns RetrievedChunk objects (schema), adapt to Document-like for wrapping
                    for rc in eresults:
                        try:
                            # anticipate rc has .text and .metadata or dict form
                            if isinstance(rc, dict):
                                page = rc.get('text', '')
                                meta = rc.get('metadata', {})
                            else:
                                page = getattr(rc, 'text', '')
                                meta = getattr(rc, 'metadata', {}) or {}
                            meta['source_type'] = 'episodic'
                            extra_memory_results.append(Document(page_content=page, metadata=meta))
                        except Exception:
                            continue
                except Exception as e:
                    print(f"Warning: could not query episodic store: {e}")

            # If policy allows persisted web_promoted, query WebPromotedStore and include results
            web_promoted_results = []
            try:
                if policy.get('web', False):
                    from CognitiveRAG.memory.web_promoted_store import WebPromotedStore
                    from CognitiveRAG.core.settings import settings
                    web_store = _injected_web_promoted_store or WebPromotedStore(getattr(settings.store, 'web_promoted_db_path', 'data/web_promoted.sqlite3'))
                    wp_results = web_store.search(query, top_k=top_k)
                    for r in wp_results:
                        try:
                            page = r.get('page_content', '')
                            meta = r.get('metadata', {})
                            meta['source_type'] = 'web_promoted'
                            meta['source_url'] = r.get('source_url')
                            web_promoted_results.append(Document(page_content=page, metadata=meta))
                        except Exception:
                            continue
            except Exception as e:
                print(f"Warning: could not query web_promoted store: {e}")

            # append extra_memory_results respecting policy
            all_docs = bm25_results + vector_results + web_results + extra_memory_results + web_promoted_results
        except Exception as e:
            print(f"Warning: Could not apply retrieval mode policy: {e}")
            all_docs = bm25_results + vector_results + web_results

        # 4. Combine and de-duplicate results (use extra_memory_results included above)
        # all_docs was assembled by the mode policy block and may include extra_memory_results
        try:
            unique_docs = {doc.page_content: doc for doc in all_docs}.values()
            unique_docs_list = list(unique_docs)
        except Exception as e:
            print(f"Warning: Could not deduplicate documents: {e}")
            unique_docs_list = all_docs
        
        print(f"Retrieved {len(unique_docs_list)} unique documents ({len(bm25_results)} BM25, {len(vector_results)} vector, {len(web_results)} web).")

        # Wrap results into RetrievedChunk structures for a single authoritative contract
        wrapped_results: List[RetrievedChunk] = []
        for idx, doc in enumerate(unique_docs_list):
            try:
                meta = getattr(doc, 'metadata', {}) or {}
                # normalize metadata to canonical contract at retrieval surface
                try:
                    from .retrieval_metadata import normalize_metadata
                    meta = normalize_metadata(meta)
                except Exception:
                    pass
                wrapped = RetrievedChunk(
                    page_content=doc.page_content,
                    metadata=meta,
                    rank=(idx + 1),
                    final_score=None,
                    ranking_reason=None,
                    augmentation_decision={}
                )
                wrapped_results.append(wrapped)
            except Exception as e:
                print(f"Warning: Could not wrap document into RetrievedChunk: {e}")

        # Attach deterministic ranking metadata (thin heuristic)
        def attach_ranking_metadata(results: List[RetrievedChunk], bm25_count: int, vector_count: int, web_count: int):
            """Populate final_score, ranking_reason, and augmentation_decision deterministically.
            Simple heuristics:
            - final_score: base score from source type and position (higher is better)
            - ranking_reason: one of 'bm25', 'vector', 'web', or combos like 'bm25+vector'
            - augmentation_decision: {'used': bool, 'source': 'web'|'local'|'mixed'}
            """
            for i, rc in enumerate(results):
                # base position score: earlier items get slightly higher score
                pos_score = 1.0 / (i + 1)

                # determine source hint from metadata or position relative to counts
                source_tags = set()
                src = rc.metadata.get('source') if isinstance(rc.metadata, dict) else None
                if src:
                    # crude heuristic: if source looks like http/https, mark web
                    if isinstance(src, str) and src.startswith('http'):
                        source_tags.add('web')
                    else:
                        source_tags.add('local')

                # fallback: infer from positional buckets
                if not source_tags:
                    if i < bm25_count:
                        source_tags.add('bm25')
                    elif i < bm25_count + vector_count:
                        source_tags.add('vector')
                    else:
                        source_tags.add('web')

                # build ranking_reason
                ranking_reason = '+'.join(sorted(source_tags))

                # final_score: combine pos_score with simple source weight
                weight = 1.0
                if 'bm25' in ranking_reason:
                    weight += 0.5
                if 'vector' in ranking_reason:
                    weight += 0.3
                if 'web' in ranking_reason:
                    weight += 0.1

                # reasoning boost: small deterministic boost when source_type indicates reasoning
                try:
                    src_type = rc.metadata.get('source_type')
                except Exception:
                    src_type = None
                reasoning_boost = 0.0
                if src_type == 'reasoning':
                    reasoning_boost = 0.2
                    try:
                        reuse_count = int(rc.metadata.get('reuse_count') or 1)
                    except Exception:
                        reuse_count = 1
                    # Bounded helper only: reuse_count can nudge ranking, never dominate it.
                    reasoning_boost += min(0.08, max(0, reuse_count - 1) * 0.02)
                    # make ranking reason explicit
                    if ranking_reason:
                        ranking_reason = ranking_reason + '+reasoning'
                    else:
                        ranking_reason = 'reasoning'

                rc.final_score = round((weight * pos_score) + reasoning_boost, 6)
                rc.ranking_reason = ranking_reason

                # augmentation decision: indicate web involvement
                if 'web' in ranking_reason:
                    rc.augmentation_decision = {'used': True, 'source': 'web'}
                elif 'reasoning' in ranking_reason:
                    rc.augmentation_decision = {'used': True, 'source': 'reasoning'}
                else:
                    rc.augmentation_decision = {'used': False, 'source': 'local'}

        # Determine counts for heuristic (bm25_results, vector_results, web_results lengths)
        try:
            bm25_count = len(bm25_results)
            vector_count = len(vector_results)
            web_count = len(web_results)
        except Exception:
            bm25_count = vector_count = web_count = 0

        attach_ranking_metadata(wrapped_results, bm25_count, vector_count, web_count)

        # Helper: persist a promoted web item into durable web_promoted store (opt-in)
        def promote_web_promoted(rc, record_id: str | None = None, store=None, db_path: str | None = None):
            try:
                from CognitiveRAG.memory.web_promoted_store import WebPromotedStore
                from CognitiveRAG.core.settings import settings
                if store is None:
                    if db_path is None:
                        db_path = getattr(settings.store, 'web_promoted_db_path', 'data/web_promoted.sqlite3')
                    store = WebPromotedStore(db_path)
                rid = record_id or f"wp_{abs(hash(rc.page_content))}"
                src = rc.metadata.get('source') or rc.metadata.get('source_url') or None
                store.upsert(rid, src, rc.page_content, rc.metadata)
                return rid
            except Exception as e:
                print(f"Warning: could not persist web_promoted record: {e}")
                return None

        # attach helper for external code paths
        try:
            # expose helper at module-level for explicit use: retriever.promote_web_promoted
            setattr(self, 'promote_web_promoted', promote_web_promoted)
        except Exception:
            pass

        return wrapped_results

# New small helper: assemble session context using context-window foundation
def assemble_session_context(
    session_id: str,
    fresh_tail_count: int = 20,
    budget: int = 4096,
    query: str | None = None,
    intent_family: str | None = None,
):
    """Return assembled context for a session: fresh_tail + summaries.
    This is a minimal runtime-exposable wrapper around CognitiveRAG.session_memory.context_window."""
    try:
        from CognitiveRAG.session_memory.context_window import assemble_context
        return assemble_context(
            session_id,
            fresh_tail_count=fresh_tail_count,
            budget=budget,
            query=query,
            intent_family=intent_family,
        )
    except Exception as e:
        print(f"Warning: could not assemble session context: {e}")
        return {"fresh_tail": [], "summaries": []}

# Singleton instance - only create if kb is available
try:
    if kb is not None:
        retriever = HybridRetriever(kb)
    else:
        retriever = None
except Exception as e:
    print(f"Warning: Could not initialize retriever: {e}")
    retriever = None
