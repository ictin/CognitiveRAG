"""Tiny OpenClaw-side shim to emit CognitiveRAG mirror events."""
import requests
from typing import Dict, Any

def emit_cognitiverag_event(event: Dict[str, Any], base_url: str = 'http://localhost:8000', timeout: int = 10) -> Dict[str, Any]:
    """Map a local event dict to CognitiveRAG mirror write surfaces.

    - Posts a user message, then an assistant message, optional part, and optional context upsert.
    - Returns a dict with the three responses (message/part/context) where present.
    """
    session_id = event.get('session_id')
    turn_id = event.get('turn_id') or event.get('message_id') or 'turn'
    user_msg_id = f"{turn_id}-user"
    assistant_msg_id = f"{turn_id}-assistant"

    # user message
    user_payload = {
        'session_id': session_id,
        'message_id': user_msg_id,
        'sender': 'user',
        'text': event.get('user_text',''),
    }
    r_user = requests.post(f"{base_url}/session_append_message", json=user_payload, timeout=timeout)
    r_user.raise_for_status()
    out = {'user': r_user.json()}

    # assistant message
    assistant_payload = {
        'session_id': session_id,
        'message_id': assistant_msg_id,
        'sender': 'assistant',
        'text': event.get('assistant_text',''),
    }
    r_assistant = requests.post(f"{base_url}/session_append_message", json=assistant_payload, timeout=timeout)
    r_assistant.raise_for_status()
    out['assistant'] = r_assistant.json()

    # optional part
    part_text = event.get('part_text')
    if part_text is not None:
        part_payload = {
            'session_id': session_id,
            'message_id': assistant_msg_id,
            'part_index': 0,
            'text': part_text,
        }
        r_part = requests.post(f"{base_url}/session_append_message_part", json=part_payload, timeout=timeout)
        r_part.raise_for_status()
        out['part'] = r_part.json()

    # optional context
    if 'context_payload' in event:
        ctx_id = event.get('context_item_id') or f"{assistant_msg_id}-ctx"
        ctx_payload = {
            'item_id': ctx_id,
            'session_id': session_id,
            'type': event.get('context_type','mirror'),
            'payload_json': event.get('context_payload') or {},
        }
        r_ctx = requests.post(f"{base_url}/session_upsert_context_item", json=ctx_payload, timeout=timeout)
        r_ctx.raise_for_status()
        out['context'] = r_ctx.json()

    return out
