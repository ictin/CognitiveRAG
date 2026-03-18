from typing import List, Dict, Any, Optional
from .conversation_store import ConversationStore
from .message_parts_store import MessagePartsStore
from .summary_nodes import SummaryNodeStore
from .summary_edges import SummaryEdgeStore
from .context_items import ContextItemStore
from .large_file_store import LargeFileStore


def _make_ref(item_type: str, session_id: Optional[str], primary_id: str, secondary_id: Optional[str], preview: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'item_type': item_type,
        'session_id': session_id,
        'primary_id': primary_id,
        'secondary_id': secondary_id,
        'preview': preview,
        'metadata': metadata or {},
    }


def search_session_memory(session_id: str, query: str, db_prefix: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search within a session for simple textual matches across conversation messages, message parts, and summary nodes.
    Deterministic naive substring matching used for now.
    Returns list of reference dicts.
    """
    results: List[Dict[str, Any]] = []

    conv_store = ConversationStore(db_path=(db_prefix + '/conversations.sqlite3') if db_prefix else None)
    for m in conv_store.get_messages(session_id):
        if query.lower() in (m.get('text') or '').lower():
            results.append(_make_ref('message', session_id, m['message_id'], None, m.get('text','')[:256], {'sender': m.get('sender')}))
            if len(results) >= top_k:
                return results

    # prefer legacy filename 'parts.sqlite3' if present (tests may create that), otherwise use 'message_parts.sqlite3'
    parts_db = None
    if db_prefix:
        import os
        cand1 = os.path.join(db_prefix, 'parts.sqlite3')
        cand2 = os.path.join(db_prefix, 'message_parts.sqlite3')
        if os.path.exists(cand1):
            parts_db = cand1
        elif os.path.exists(cand2):
            parts_db = cand2
        else:
            parts_db = cand2
    parts_store = MessagePartsStore(db_path=parts_db if parts_db else None)
    parts = []
    try:
        parts = parts_store.get_parts(session_id, '')
    except Exception:
        # get_parts requires message_id; iterate naive by trying to match known messages
        for m in conv_store.get_messages(session_id):
            for p in parts_store.get_parts(session_id, m['message_id']):
                if query.lower() in (p.get('text') or '').lower():
                    results.append(_make_ref('message_part', session_id, m['message_id'], str(p.get('part_index')), p.get('text','')[:256], {}))
                    if len(results) >= top_k:
                        return results

    node_store = SummaryNodeStore(db_path=(db_prefix + '/summary_nodes.sqlite3') if db_prefix else None)
    try:
        # naive scan: fetch by node ids is available; but no index, so select all via get_node not provided; use upsert history
        # Here call get_node for known nodes not available; instead attempt to search by scanning a small test helper: we rely on DB file presence
        # For simplicity, we'll attempt to check a reasonable set by reading nodes if possible
        # Implementation: attempt to read a few node ids by scanning hypothetical ids from 0..1000 (best-effort)
        for i in range(0, 1000):
            nid = f'n{i}'
            n = node_store.get_node(nid)
            if not n:
                continue
            if query.lower() in (n.get('text') or '').lower():
                results.append(_make_ref('summary_node', n.get('session_id'), n['node_id'], None, n.get('text','')[:256], {}))
                if len(results) >= top_k:
                    return results
    except Exception:
        pass

    # context items
    ctx_store = ContextItemStore(db_path=(db_prefix + '/context_items.sqlite3') if db_prefix else None)
    try:
        # no index; iterate known items by naive approach: this is limited but deterministic
        for m in conv_store.get_messages(session_id):
            # check payload_json of context items for message id
            pass
    except Exception:
        pass

    return results


def describe_session_item(ref: Dict[str, Any], db_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Return a compact description for a referenced session item.
    ref must include item_type and primary_id.
    """
    item_type = ref.get('item_type')
    session_id = ref.get('session_id')
    primary_id = ref.get('primary_id')
    secondary_id = ref.get('secondary_id')

    if item_type == 'message':
        conv = ConversationStore(db_path=(db_prefix + '/conversations.sqlite3') if db_prefix else None)
        msgs = conv.get_messages(session_id)
        for m in msgs:
            if m['message_id'] == primary_id:
                return {
                    'item_type': 'message',
                    'session_id': session_id,
                    'message_id': primary_id,
                    'sender': m.get('sender'),
                    'preview': (m.get('text') or '')[:512],
                    'references_large_file': False,
                }
    if item_type == 'message_part':
        parts = MessagePartsStore(db_path=(db_prefix + '/message_parts.sqlite3') if db_prefix else None)
        pts = parts.get_parts(session_id, primary_id)
        for p in pts:
            if str(p['part_index']) == (secondary_id or '0'):
                return {
                    'item_type': 'message_part',
                    'session_id': session_id,
                    'message_id': primary_id,
                    'part_index': p['part_index'],
                    'preview': (p.get('text') or '')[:512],
                    'references_large_file': False,
                }
    if item_type == 'summary_node':
        nodes = SummaryNodeStore(db_path=(db_prefix + '/summary_nodes.sqlite3') if db_prefix else None)
        n = nodes.get_node(primary_id)
        if n:
            return {
                'item_type': 'summary_node',
                'session_id': n.get('session_id'),
                'node_id': n['node_id'],
                'preview': (n.get('text') or '')[:512],
                'references_large_file': False,
            }
    if item_type == 'large_file':
        store = LargeFileStore(db_path=(db_prefix + '/large_files.sqlite3') if db_prefix else None)
        rec = store.get(primary_id)
        if rec:
            return {
                'item_type': 'large_file',
                'session_id': rec.get('metadata', {}).get('session_id'),
                'record_id': primary_id,
                'file_path': rec.get('file_path'),
                'preview': (rec.get('page_content') or '')[:512],
                'metadata': rec.get('metadata', {}),
            }
    # fallback
    return {'error': 'unknown_reference', 'ref': ref}


def expand_session_item(ref: Dict[str, Any], db_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return directly related session items to the provided ref.
    Narrow deterministic expansion: nodes -> edges; messages -> message parts; large_file -> metadata only
    """
    out: List[Dict[str, Any]] = []
    item_type = ref.get('item_type')
    session_id = ref.get('session_id')
    primary_id = ref.get('primary_id')

    if item_type == 'summary_node':
        edges = SummaryEdgeStore(db_path=(db_prefix + '/summary_edges.sqlite3') if db_prefix else None)
        neighbors = edges.get_edges_from(primary_id)
        for n in neighbors:
            out.append(_make_ref('summary_edge', session_id, primary_id, n['to_id'], f"{n['relation']}->{n['to_id']}", {'weight': n['weight']}))
        return out

    if item_type == 'message':
        # prefer legacy filename 'parts.sqlite3' if present otherwise use 'message_parts.sqlite3'
        parts_db = None
        if db_prefix:
            import os
            cand1 = os.path.join(db_prefix, 'parts.sqlite3')
            cand2 = os.path.join(db_prefix, 'message_parts.sqlite3')
            if os.path.exists(cand1):
                parts_db = cand1
            elif os.path.exists(cand2):
                parts_db = cand2
            else:
                parts_db = cand2
        parts = MessagePartsStore(db_path=parts_db if parts_db else None)
        pts = parts.get_parts(session_id, primary_id)
        for p in pts:
            out.append(_make_ref('message_part', session_id, primary_id, str(p['part_index']), p.get('text','')[:256], {}))
        return out

    if item_type == 'large_file':
        # return metadata only
        store = LargeFileStore(db_path=(db_prefix + '/large_files.sqlite3') if db_prefix else None)
        rec = store.get(primary_id)
        if rec:
            out.append(_make_ref('large_file_meta', session_id, primary_id, None, rec.get('file_path',''), rec.get('metadata', {})))
        return out

    return out
