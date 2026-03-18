from __future__ import annotations

import requests
from typing import Any
import logging
import json

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

    def _normalize_content(self, value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                if isinstance(item, dict) and 'text' in item:
                    parts.append(str(item['text']))
                else:
                    parts.append(str(item))
            return '\n\n'.join(parts)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def chat(self, model: str, messages: list[dict], max_tokens: int = 512, stream: bool = False) -> dict:
        url = f"{self.base_url}{self.api_path}/chat"
        # sanitize messages
        normalized = []
        for m in messages:
            role = str(m.get('role', 'user'))
            content = self._normalize_content(m.get('content', ''))
            normalized.append({'role': role, 'content': content})
        payload = {"model": model, "messages": normalized, "max_tokens": max_tokens, "stream": stream}
        # logging per-message summary
        logging.getLogger().info('OLLAMA_PAYLOAD: url=%s model=%s messages=%d', url, model, len(normalized))
        for i, m in enumerate(normalized):
            logging.getLogger().info('OLLAMA_MSG %d: role=%s type=%s preview=%s', i, m['role'], type(m['content']).__name__, m['content'][:200])
        logging.getLogger().info('OLLAMA_FULL_PAYLOAD: %s', json.dumps(payload, ensure_ascii=False)[:2000])

        r = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        if r.status_code >= 400:
            logging.getLogger().error('OLLAMA_ERROR: status=%s body=%s', r.status_code, r.text)
        r.raise_for_status()
        try:
            data = r.json()
        except Exception:
            # Ollama may return multiple JSON objects or streaming responses; try to parse first JSON object from text
            txt = r.text
            lines = [l for l in txt.splitlines() if l.strip()]
            try:
                data = json.loads(lines[0]) if lines else txt
            except Exception:
                data = txt
        return data


