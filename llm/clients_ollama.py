from __future__ import annotations

import requests
from typing import Any

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

    def chat(self, model: str, messages: list[dict], max_tokens: int = 512, stream: bool = False) -> dict:
        url = f"{self.base_url}{self.api_path}/chat"
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()
