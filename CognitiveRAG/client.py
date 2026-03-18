"""Minimal OpenClaw-facing client helper for CognitiveRAG HTTP endpoints.
Provides two functions:
- query(session_id, query, base_url)
- promote_session(session_id, base_url)

This is intentionally tiny and synchronous (wraps requests) for easy integration.
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
