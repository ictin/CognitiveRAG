from __future__ import annotations

import re


_PREFIX_PATTERNS = [
    r"^\s*the user said (that )?",
    r"^\s*the user mentioned (that )?",
    r"^\s*i remember (that )?",
    r"^\s*we concluded (that )?",
    r"^\s*it was concluded (that )?",
    r"^\s*we learned (that )?",
    r"^\s*note that\s*",
]


def strip_speech_act(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    for pattern in _PREFIX_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*[:\-]\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

