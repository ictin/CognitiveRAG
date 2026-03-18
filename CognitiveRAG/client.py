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
