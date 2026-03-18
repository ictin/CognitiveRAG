from typing import Dict, Any
from datetime import datetime

CANONICAL_KEYS = [
    'source_type',
    'project',
    'memory_scope',
    'document_kind',
    'component',
    'topic_tags',
    'entity_tags',
    'environment_tags',
    'importance',
    'confidence',
    'origin_id',
    'parent_id',
    'test_run_id',
    'created_at',
    'updated_at',
]


def normalize_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a metadata dict that contains canonical keys where possible.
    This is a lightweight normalization used at retrieval surface; it does not mutate storage.
    """
    out: Dict[str, Any] = {}
    if not isinstance(raw, dict):
        raw = {}

    # source_type: prefer explicit keys 'source_type' or 'source'
    src = raw.get('source_type') or raw.get('source') or raw.get('source_type')
    if isinstance(src, str):
        out['source_type'] = src
    else:
        out['source_type'] = raw.get('source_type', 'unknown')

    out['project'] = raw.get('project') or raw.get('project_id') or 'cognitiverag'
    out['memory_scope'] = raw.get('memory_scope') or raw.get('scope') or None
    out['document_kind'] = raw.get('document_kind') or raw.get('kind') or None
    out['component'] = raw.get('component') or None

    # tags: ensure list shape
    def to_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, str):
            return [v]
        return list(v)

    out['topic_tags'] = to_list(raw.get('topic_tags') or raw.get('topics') or [])
    out['entity_tags'] = to_list(raw.get('entity_tags') or raw.get('entities') or [])
    out['environment_tags'] = to_list(raw.get('environment_tags') or raw.get('env') or [])

    # numeric fields
    out['importance'] = float(raw.get('importance', 0.0)) if raw.get('importance') is not None else 0.0
    out['confidence'] = float(raw.get('confidence', 0.0)) if raw.get('confidence') is not None else 0.0

    out['origin_id'] = raw.get('origin_id') or raw.get('chunk_id') or raw.get('document_id') or None
    out['parent_id'] = raw.get('parent_id') or None
    out['test_run_id'] = raw.get('test_run_id') or None

    # timestamps
    now = datetime.utcnow().isoformat()
    out['created_at'] = raw.get('created_at') or raw.get('created') or now
    out['updated_at'] = raw.get('updated_at') or raw.get('updated') or now

    # If the raw dict contains a nested 'metadata' dict (common in stores), merge simple keys up
    nested = raw.get('metadata') if isinstance(raw, dict) else None
    if isinstance(nested, dict):
        for k, v in nested.items():
            if k not in out:
                out[k] = v

    # Also merge top-level non-canonical keys from the raw metadata dict (e.g., {'injected': True})
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k not in out and k != 'metadata':
                out[k] = v

    # keep any other useful keys for backward compatibility
    out['_raw'] = raw
    return out
