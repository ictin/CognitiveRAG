"""Minimal OpenClaw-facing client helper for CognitiveRAG HTTP endpoints.
Provides two functions:
- query(session_id, query, base_url)
- promote_session(session_id, base_url)
- append_message(session_id, message_id, sender, text, created_at=None)
"""
from __future__ import annotations
import requests
from typing import Optional, Dict, Any


DEFAULT_BASE = "http://localhost:8000"


def query(session_id: Optional[str], query_text: str, base_url: str = DEFAULT_BASE, timeout: int = 10) -> Dict[str, Any]:
    payload = {"query": query_text}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(f"{base_url}/query", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def promote_session(session_id: str, base_url: str = DEFAULT_BASE, timeout: int = 10) -> Dict[str, Any]:
    payload = {"session_id": session_id}
    resp = requests.post(f"{base_url}/promote_session", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def append_message(session_id: str, message_id: str, sender: str, text: str, created_at: Optional[str] = None, base_url: str = DEFAULT_BASE, timeout: int = 10) -> Dict[str, Any]:
    payload = {
        'session_id': session_id,
        'message_id': message_id,
        'sender': sender,
        'text': text,
    }
    if created_at:
        payload['created_at'] = created_at
    resp = requests.post(f"{base_url}/session_append_message", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def append_message_part(session_id: str, message_id: str, part_index: int, text: str, meta_json: Optional[dict] = None, base_url: str = DEFAULT_BASE, timeout: int = 10) -> Dict[str, Any]:
    payload = {
        'session_id': session_id,
        'message_id': message_id,
        'part_index': part_index,
        'text': text,
    }
    if meta_json is not None:
        payload['meta_json'] = meta_json
    resp = requests.post(f"{base_url}/session_append_message_part", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def upsert_context_item(item_id: str, session_id: str, type: str, payload_json: dict, created_at: Optional[str] = None, base_url: str = DEFAULT_BASE, timeout: int = 10) -> Dict[str, Any]:
    payload = {
        'item_id': item_id,
        'session_id': session_id,
        'type': type,
        'payload_json': payload_json,
    }
    if created_at is not None:
        payload['created_at'] = created_at
    resp = requests.post(f"{base_url}/session_upsert_context_item", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def mirror_write_interaction(session_id: str, message_id: str, sender: str, text: str, part_text: str, context_item_id: str, context_payload: dict, base_url: str = DEFAULT_BASE, timeout: int = 10) -> Dict[str, Dict[str, Any]]:
    """Mirror-write adapter: perform one message, one part, and one context upsert.
    Parameters are simple primitives so OpenClaw callers can invoke this in one call.
    Returns a dict with responses for 'message', 'part', and 'context'.

    Example:
        from CognitiveRAG import client as cl
        res = cl.mirror_write_interaction(
            session_id='s1', message_id='m1', sender='assistant',
            text='Answer', part_text='chunk-0', context_item_id='ctx1',
            context_payload={'k':'v'}
        )
        # res['message'], res['part'], res['context'] available
    """
    """Perform a single mirror-write interaction:
    - append_message
    - append_message_part (part_index=0)
    - upsert_context_item

    Returns a dict with responses from each call.
    """
    res_message = append_message(session_id, message_id, sender, text, base_url=base_url, timeout=timeout)
    res_part = append_message_part(session_id, message_id, 0, part_text, base_url=base_url, timeout=timeout)
    res_context = upsert_context_item(context_item_id, session_id, 'mirror', context_payload, base_url=base_url, timeout=timeout)
    return {
        'message': res_message,
        'part': res_part,
        'context': res_context,
    }


def mirror_turn(session_id: str, turn_id: str, user_text: str, assistant_text: str, part_text: Optional[str] = None, context_item_id: Optional[str] = None, context_payload: Optional[dict] = None, base_url: str = DEFAULT_BASE, timeout: int = 10) -> Dict[str, Dict[str, Any]]:
    """Higher-level helper: mirror a single user->assistant turn.

    This packages two messages (user then assistant), optional part for the
    assistant message, and an optional context item. It delegates to the
    existing lower-level mirror_write_interaction path for the assistant
    message and uses append_message for the user message.

    Parameters:
    - session_id, turn_id: identifiers. turn_id is used to derive message_ids.
    - user_text, assistant_text: message contents.
    - part_text: optional assistant part text (sent as part_index=0)
    - context_item_id/context_payload: optional context upsert for the assistant

    Returns a dict with keys 'user', 'assistant' and optional 'context'.
    """
    user_msg_id = f"{turn_id}-user"
    assistant_msg_id = f"{turn_id}-assistant"

    # write user message
    res_user = append_message(session_id, user_msg_id, 'user', user_text, base_url=base_url, timeout=timeout)

    # write assistant message + optional part + optional context
    if part_text is None and context_item_id is None:
        # simple assistant message without parts/context
        res_assistant = append_message(session_id, assistant_msg_id, 'assistant', assistant_text, base_url=base_url, timeout=timeout)
        return {'user': res_user, 'assistant': res_assistant}

    # use mirror_write_interaction to handle assistant message, part, and context
    res = mirror_write_interaction(
        session_id=session_id,
        message_id=assistant_msg_id,
        sender='assistant',
        text=assistant_text,
        part_text=(part_text or ''),
        context_item_id=(context_item_id or f"{assistant_msg_id}-ctx"),
        context_payload=(context_payload or {}),
        base_url=base_url,
        timeout=timeout,
    )
    # combine responses
    out = {'user': res_user, 'assistant': res['message'], 'part': res['part'], 'context': res['context']}
    return out


def mirror_agent_event(event: dict, base_url: str = DEFAULT_BASE, timeout: int = 10) -> Dict[str, Any]:
    """Shim for OpenClaw-style agent events.

    Expected event keys (example):
      {
        'session_id': 's1',
        'turn_id': 't1',
        'user_text': 'Hello',
        'assistant_text': 'Hi',
        'part_text': 'chunk-0',           # optional
        'context_payload': {'k':'v'},     # optional
        'context_item_id': 'ctx-1'        # optional
      }

    This maps the event to mirror_turn(...) and returns the mirror_turn result.
    """
    session_id = event.get('session_id')
    turn_id = event.get('turn_id') or event.get('message_id') or 'turn'
    user_text = event.get('user_text','')
    assistant_text = event.get('assistant_text','')
    part_text = event.get('part_text')
    ctx_id = event.get('context_item_id')
    ctx_payload = event.get('context_payload')

    return mirror_turn(
        session_id=session_id,
        turn_id=turn_id,
        user_text=user_text,
        assistant_text=assistant_text,
        part_text=part_text,
        context_item_id=ctx_id,
        context_payload=ctx_payload,
        base_url=base_url,
        timeout=timeout,
    )
