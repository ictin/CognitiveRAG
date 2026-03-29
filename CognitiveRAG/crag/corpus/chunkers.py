from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


_MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".rst"}
_CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".swift", ".kt", ".kts",
}
_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_CODE_SYMBOL_RE = re.compile(
    r"(?m)^(?:\s*)(?:def|class|function|interface|enum|struct|impl|func)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass
class CorpusChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any]


def _make_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}_chunk_{index:04d}"


def _detect_source_type(source_path: str, text: str, chunk_size: int) -> str:
    suffix = Path(source_path).suffix.lower()
    if suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in _CODE_SUFFIXES:
        return "code"
    if len(text) > max(chunk_size * 8, 20_000):
        return "large_file"
    return "text"


def _line_for_offset(text: str, offset: int) -> int:
    off = max(0, min(len(text), int(offset)))
    return text.count("\n", 0, off) + 1


def _window_with_overlap(length: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    step = max(1, int(chunk_size) - int(chunk_overlap))
    while start < length:
        end = min(length, start + int(chunk_size))
        ranges.append((start, end))
        if end >= length:
            break
        start += step
    return ranges


def _align_end_to_boundary(text: str, start: int, end: int, search_back: int = 240) -> int:
    if end >= len(text):
        return len(text)
    window_start = max(start + 1, end - search_back)
    snippet = text[window_start:end]
    para = snippet.rfind("\n\n")
    if para >= 0:
        return window_start + para + 2
    line = snippet.rfind("\n")
    if line >= 0:
        return window_start + line + 1
    return end


def _section_spans_markdown(text: str) -> List[Dict[str, Any]]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [{"title": "document", "start": 0, "end": len(text), "level": 0}]

    sections: List[Dict[str, Any]] = []
    if matches[0].start() > 0:
        sections.append({"title": "preamble", "start": 0, "end": matches[0].start(), "level": 0})

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(
            {
                "title": m.group(2).strip(),
                "start": start,
                "end": end,
                "level": len(m.group(1)),
            }
        )
    return sections


def _chunk_markdown(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    sections = _section_spans_markdown(text)
    chunks: List[Dict[str, Any]] = []
    min_merge_len = max(120, int(chunk_size * 0.30))
    pending: Dict[str, Any] | None = None

    for idx, section in enumerate(sections):
        if pending is not None:
            section = {
                "title": pending["title"],
                "start": pending["start"],
                "end": section["end"],
                "level": pending.get("level", 0),
            }
            pending = None

        section_len = section["end"] - section["start"]
        if section_len < min_merge_len and idx + 1 < len(sections) and int(section.get("level", 0)) == 0:
            pending = section
            continue

        ranges = _window_with_overlap(section_len, chunk_size, chunk_overlap)
        for local_start, local_end in ranges:
            abs_start = section["start"] + local_start
            abs_end = section["start"] + _align_end_to_boundary(
                text=text,
                start=section["start"] + local_start,
                end=section["start"] + local_end,
            )
            if abs_end <= abs_start:
                abs_end = section["start"] + local_end
            if abs_end <= abs_start:
                continue
            chunk_text = text[abs_start:abs_end].strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "text": chunk_text,
                    "char_start": abs_start,
                    "char_end": abs_end,
                    "strategy": "markdown_section",
                    "section_title": section["title"],
                    "section_level": section["level"],
                }
            )

    if pending is not None:
        chunk_text = text[pending["start"]:pending["end"]].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "char_start": pending["start"],
                    "char_end": pending["end"],
                    "strategy": "markdown_section",
                    "section_title": pending["title"],
                    "section_level": pending.get("level", 0),
                }
            )

    return chunks


def _symbol_spans_code(text: str) -> List[Dict[str, Any]]:
    matches = list(_CODE_SYMBOL_RE.finditer(text))
    if not matches:
        return [{"name": "module", "start": 0, "end": len(text)}]
    spans: List[Dict[str, Any]] = []
    if matches[0].start() > 0:
        spans.append({"name": "module_preamble", "start": 0, "end": matches[0].start()})
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append({"name": m.group(1), "start": start, "end": end})
    return spans


def _chunk_code(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    spans = _symbol_spans_code(text)
    chunks: List[Dict[str, Any]] = []
    for span in spans:
        span_len = span["end"] - span["start"]
        ranges = _window_with_overlap(span_len, chunk_size, chunk_overlap) if span_len > chunk_size else [(0, span_len)]
        for local_start, local_end in ranges:
            abs_start = span["start"] + local_start
            abs_end = span["start"] + local_end
            chunk_text = text[abs_start:abs_end].strip("\n")
            if not chunk_text.strip():
                continue
            chunks.append(
                {
                    "text": chunk_text,
                    "char_start": abs_start,
                    "char_end": abs_end,
                    "strategy": "code_symbol",
                    "symbol_name": span["name"],
                }
            )
    return chunks


def _chunk_large_file(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    ranges = _window_with_overlap(len(text), max(chunk_size, 1200), chunk_overlap)
    total = len(ranges)
    chunks: List[Dict[str, Any]] = []
    for i, (start, end) in enumerate(ranges):
        excerpt = text[start:end].strip()
        if not excerpt:
            continue
        chunks.append(
            {
                "text": excerpt,
                "char_start": start,
                "char_end": end,
                "strategy": "large_file_excerpt",
                "excerpt_window_index": i,
                "excerpt_window_total": total,
                "document_kind": "large_file_excerpt",
            }
        )
    return chunks


def _chunk_generic_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    ranges = _window_with_overlap(len(text), chunk_size, chunk_overlap)
    chunks: List[Dict[str, Any]] = []
    for start, end in ranges:
        aligned_end = _align_end_to_boundary(text, start, end)
        if aligned_end <= start:
            aligned_end = end
        piece = text[start:aligned_end].strip()
        if not piece:
            continue
        chunks.append(
            {
                "text": piece,
                "char_start": start,
                "char_end": aligned_end,
                "strategy": "generic_window",
            }
        )
    return chunks


def _with_overlap_metadata(base_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(base_chunks):
        prev = base_chunks[i - 1] if i > 0 else None
        nxt = base_chunks[i + 1] if i + 1 < len(base_chunks) else None
        overlap_prev = 0
        overlap_next = 0
        if prev is not None:
            overlap_prev = max(0, prev["char_end"] - item["char_start"])
        if nxt is not None:
            overlap_next = max(0, item["char_end"] - nxt["char_start"])
        enriched = dict(item)
        enriched["overlap_prev_chars"] = overlap_prev
        enriched["overlap_next_chars"] = overlap_next
        out.append(enriched)
    return out


def chunk_document(
    *,
    document_id: str,
    source_path: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    base_metadata: Dict[str, Any] | None = None,
) -> List[CorpusChunk]:
    source_type = _detect_source_type(source_path, text, chunk_size)
    if source_type == "markdown":
        raw_chunks = _chunk_markdown(text, chunk_size, chunk_overlap)
    elif source_type == "code":
        raw_chunks = _chunk_code(text, chunk_size, chunk_overlap)
    elif source_type == "large_file":
        raw_chunks = _chunk_large_file(text, chunk_size, chunk_overlap)
    else:
        raw_chunks = _chunk_generic_text(text, chunk_size, chunk_overlap)

    enriched_chunks = _with_overlap_metadata(raw_chunks)
    suffix = Path(source_path).suffix.lower()
    out: List[CorpusChunk] = []
    for idx, c in enumerate(enriched_chunks):
        md = dict(base_metadata or {})
        md.update(
            {
                "source_path": source_path,
                "source_suffix": suffix,
                "source_format": source_type,
                "chunk_index": idx,
                "chunk_char_start": int(c["char_start"]),
                "chunk_char_end": int(c["char_end"]),
                "chunk_line_start": _line_for_offset(text, int(c["char_start"])),
                "chunk_line_end": _line_for_offset(text, int(c["char_end"])),
                "overlap_prev_chars": int(c.get("overlap_prev_chars", 0)),
                "overlap_next_chars": int(c.get("overlap_next_chars", 0)),
                "chunk_strategy": c.get("strategy", "generic_window"),
                "section_title": c.get("section_title"),
                "section_level": c.get("section_level"),
                "symbol_name": c.get("symbol_name"),
                "excerpt_window_index": c.get("excerpt_window_index"),
                "excerpt_window_total": c.get("excerpt_window_total"),
                "document_kind": c.get("document_kind", "corpus_chunk"),
                "provenance": {
                    "source_path": source_path,
                    "source_format": source_type,
                    "char_start": int(c["char_start"]),
                    "char_end": int(c["char_end"]),
                    "line_start": _line_for_offset(text, int(c["char_start"])),
                    "line_end": _line_for_offset(text, int(c["char_end"])),
                },
            }
        )
        out.append(
            CorpusChunk(
                chunk_id=_make_chunk_id(document_id, idx),
                document_id=document_id,
                chunk_index=idx,
                text=c["text"],
                metadata=md,
            )
        )
    return out
