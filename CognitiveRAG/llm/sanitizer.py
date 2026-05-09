from __future__ import annotations

import json
import re
from typing import List, Dict, Any

_METADATA_KEYS = {"message_id", "sender_id", "chat_id", "sender", "timestamp"}
_METADATA_HEADER_RE = re.compile(r"^\s*sender\s*\(untrusted metadata\)\s*:\s*$", re.IGNORECASE)
_METADATA_LINE_RE = re.compile(r'^\s*"?(message_id|sender_id|chat_id|sender|timestamp)"?\s*:', re.IGNORECASE)


def _contains_metadata_object_keys(raw: str) -> bool:
    try:
        parsed = json.loads(raw)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    return any(str(k).lower() in _METADATA_KEYS for k in parsed.keys())


def _strip_fenced_json_metadata(text: str) -> str:
    if not isinstance(text, str):
        return text
    fenced_re = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

    def _replace(match: re.Match) -> str:
        body = match.group(1).strip()
        return "" if _contains_metadata_object_keys(body) else match.group(0)

    return fenced_re.sub(_replace, text)


def _strip_inline_json_metadata(text: str) -> str:
    if not isinstance(text, str):
        return text
    inline_re = re.compile(r"\{[^{}]{1,3000}\}")

    def _replace(match: re.Match) -> str:
        block = match.group(0)
        return "" if _contains_metadata_object_keys(block) else block

    return inline_re.sub(_replace, text)


def sanitize_text(text: Any) -> str:
    if not isinstance(text, str):
        return str(text)
    lines = text.splitlines()
    kept: list[str] = []
    skip_until_blank = False
    for ln in lines:
        if _METADATA_HEADER_RE.match(ln):
            skip_until_blank = True
            continue
        if skip_until_blank:
            if not ln.strip():
                skip_until_blank = False
            continue
        if _METADATA_LINE_RE.match(ln):
            continue
        kept.append(ln)

    cleaned = "\n".join(kept).strip()
    cleaned = _strip_fenced_json_metadata(cleaned)
    cleaned = _strip_inline_json_metadata(cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned).strip()
    return cleaned


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
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
        sanitized = sanitize_text(content_str)
        out.append({"role": role, "content": sanitized})
    return out
