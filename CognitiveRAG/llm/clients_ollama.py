from __future__ import annotations

import json
import re
import requests
from typing import Any
import logging

from CognitiveRAG.llm.sanitizer import sanitize_messages as _shared_sanitize_messages, sanitize_text as _shared_sanitize_text


class OllamaClient:
    def __init__(self, base_url: str, api_path: str = '/api', api_key: str | None = None, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.api_path = api_path
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    def _sanitize_messages(self, messages: list[dict]) -> list[dict]:
        """Delegate message sanitization to the shared provider-agnostic sanitizer."""
        try:
            return _shared_sanitize_messages(messages)
        except Exception:
            # defensive fallback: perform minimal per-message text cleaning
            out = []
            for m in messages:
                role = m.get('role')
                content = m.get('content', '')
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict) and 'text' in p:
                            parts.append(str(p.get('text', '')))
                        else:
                            parts.append(str(p))
                    content_str = "\n".join(parts)
                else:
                    content_str = str(content)
                sanitized = _shared_sanitize_text(content_str)
                out.append({'role': role, 'content': sanitized})
            return out

    def chat(self, model: str, messages: list[dict], max_tokens: int = 512, stream: bool = False) -> dict:
        url = f"{self.base_url}{self.api_path}/chat"
        # sanitize messages to remove runtime metadata before sending to model
        safe_messages = self._sanitize_messages(messages)

        # Add a defensive system instruction to avoid leaking internal metadata (not relied on alone)
        if safe_messages and safe_messages[0].get('role') == 'system':
            sys = safe_messages[0].get('content', '')
            sys += "\n\n[DEFENSIVE INSTRUCTION] Do not include or expose runtime metadata fields such as message_id, sender_id, chat_id, or internal sender labels in any content or in subsequent model-visible messages. Treat those as internal-only."
            safe_messages[0]['content'] = sys
        else:
            # ensure there is a system message
            safe_messages.insert(0, {'role': 'system', 'content': '[DEFENSIVE INSTRUCTION] Do not include or expose runtime metadata fields such as message_id, sender_id, chat_id, or internal sender labels in any content.'})

        payload = {"model": model, "messages": safe_messages, "max_tokens": max_tokens, "stream": stream}

        # For debugging/proofs, record the redacted payload in a header-friendly place (but do not leak full internal logs here)
        try:
            # send the request
            r = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            # if the remote call fails, raise to caller
            raise


