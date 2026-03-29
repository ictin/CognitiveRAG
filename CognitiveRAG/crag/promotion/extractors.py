from __future__ import annotations

import re
from typing import List

from CognitiveRAG.crag.promotion.speech_act import strip_speech_act


def _sentences(text: str) -> list[str]:
    raw = re.split(r"[.\n]+", str(text or ""))
    out: list[str] = []
    for piece in raw:
        s = strip_speech_act(piece)
        if s:
            out.append(s)
    return out


def extract_propositions(text: str) -> List[str]:
    out: list[str] = []
    for sentence in _sentences(text):
        s = sentence.lower()
        if any(
            key in s
            for key in [
                "user prefers",
                "user likes",
                "user wants",
                "preference",
                "project uses",
                "project is",
                "environment",
                "constraint",
            ]
        ):
            out.append(sentence)
    if not out:
        first = _sentences(text)[:1]
        out.extend(first)
    return out


def extract_prescriptions(text: str) -> List[str]:
    out: list[str] = []
    for sentence in _sentences(text):
        s = sentence.lower()
        if (
            "workflow" in s
            or "procedure" in s
            or "run " in s
            or "then " in s
            or "step " in s
            or " -> " in sentence
            or "→" in sentence
            or "successful" in s
        ):
            out.append(sentence)
    return out

