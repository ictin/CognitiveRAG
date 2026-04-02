from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from difflib import SequenceMatcher

from CognitiveRAG.schemas.memory import ReasoningPattern
import json


class ReasoningStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reasoning_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    problem_signature TEXT NOT NULL,
                    reasoning_steps_json TEXT NOT NULL,
                    solution_summary TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    provenance_json TEXT,
                    memory_subtype TEXT,
                    normalized_text TEXT,
                    freshness_state TEXT,
                    exact_fingerprint TEXT,
                    near_fingerprint TEXT,
                    canonical_pattern_id TEXT,
                    near_duplicate_of TEXT,
                    reuse_count INTEGER NOT NULL DEFAULT 1,
                    merged_from_json TEXT
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(reasoning_patterns)").fetchall()}
            if "memory_subtype" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN memory_subtype TEXT")
            if "normalized_text" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN normalized_text TEXT")
            if "freshness_state" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN freshness_state TEXT")
            if "exact_fingerprint" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN exact_fingerprint TEXT")
            if "near_fingerprint" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN near_fingerprint TEXT")
            if "canonical_pattern_id" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN canonical_pattern_id TEXT")
            if "near_duplicate_of" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN near_duplicate_of TEXT")
            if "reuse_count" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN reuse_count INTEGER NOT NULL DEFAULT 1")
            if "merged_from_json" not in cols:
                conn.execute("ALTER TABLE reasoning_patterns ADD COLUMN merged_from_json TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_exact_fp ON reasoning_patterns(exact_fingerprint)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_near_fp ON reasoning_patterns(near_fingerprint)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_canonical ON reasoning_patterns(canonical_pattern_id)")

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        raw = (text or "").strip().lower()
        raw = re.sub(r"\s+", " ", raw)
        return raw

    @classmethod
    def _normalize_signature(cls, signature: str | None) -> str:
        return cls._normalize_text(signature)

    @classmethod
    def _exact_fingerprint(cls, *, signature: str, normalized_text: str, memory_subtype: str | None) -> str:
        seed = f"{cls._normalize_signature(signature)}|{cls._normalize_text(normalized_text)}|{cls._normalize_text(memory_subtype)}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()

    @classmethod
    def _near_fingerprint(cls, *, signature: str, normalized_text: str, memory_subtype: str | None) -> str:
        seed_parts = f"{cls._normalize_signature(signature)} {cls._normalize_text(normalized_text)} {cls._normalize_text(memory_subtype)}"
        tokens = sorted({tok for tok in re.split(r"[^a-z0-9]+", seed_parts) if tok})
        compact = " ".join(tokens[:24])
        return hashlib.sha1(compact.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_json_list(payload: str | None) -> list:
        if not payload:
            return []
        try:
            obj = json.loads(payload)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    @staticmethod
    def _merge_unique(existing: list, incoming: list) -> list:
        out = list(existing or [])
        for item in incoming or []:
            if item not in out:
                out.append(item)
        return out

    @classmethod
    def _similarity(cls, *, a_sig: str, a_text: str, b_sig: str, b_text: str) -> float:
        left = f"{cls._normalize_signature(a_sig)} | {cls._normalize_text(a_text)}"
        right = f"{cls._normalize_signature(b_sig)} | {cls._normalize_text(b_text)}"
        return SequenceMatcher(None, left, right).ratio()

    def upsert(self, pattern: ReasoningPattern) -> None:
        incoming_signature = pattern.problem_signature or ""
        incoming_text = pattern.normalized_text or pattern.solution_summary or ""
        incoming_subtype = getattr(pattern, "memory_subtype", None)
        exact_fp = self._exact_fingerprint(
            signature=incoming_signature,
            normalized_text=incoming_text,
            memory_subtype=incoming_subtype,
        )
        near_fp = self._near_fingerprint(
            signature=incoming_signature,
            normalized_text=incoming_text,
            memory_subtype=incoming_subtype,
        )
        incoming_provenance = list(pattern.provenance or [])

        with self._connect() as conn:
            canonical = conn.execute(
                """
                SELECT pattern_id, confidence, provenance_json, reuse_count, merged_from_json
                FROM reasoning_patterns
                WHERE exact_fingerprint=?
                ORDER BY reuse_count DESC, pattern_id ASC
                LIMIT 1
                """,
                (exact_fp,),
            ).fetchone()

            # Exact duplicate: keep one canonical record and increment reuse count.
            if canonical:
                merged_provenance = self._merge_unique(self._parse_json_list(canonical[2]), incoming_provenance)
                merged_from = self._merge_unique(self._parse_json_list(canonical[4]), [pattern.pattern_id])
                conn.execute(
                    """
                    UPDATE reasoning_patterns
                    SET confidence=?,
                        provenance_json=?,
                        reuse_count=?,
                        merged_from_json=?,
                        memory_subtype=?,
                        normalized_text=?,
                        freshness_state=?,
                        exact_fingerprint=?,
                        near_fingerprint=?,
                        canonical_pattern_id=?
                    WHERE pattern_id=?
                    """,
                    (
                        max(float(canonical[1] or 0.0), float(pattern.confidence or 0.0)),
                        json.dumps(merged_provenance),
                        int(canonical[3] or 1) + 1,
                        json.dumps(merged_from),
                        incoming_subtype,
                        incoming_text,
                        getattr(pattern, "freshness_state", None),
                        exact_fp,
                        near_fp,
                        canonical[0],
                        canonical[0],
                    ),
                )
                return

            existing = conn.execute(
                """
                SELECT pattern_id, confidence, provenance_json, reuse_count, merged_from_json
                FROM reasoning_patterns
                WHERE pattern_id=?
                LIMIT 1
                """,
                (pattern.pattern_id,),
            ).fetchone()
            existing_reuse = int(existing[3] or 1) if existing else 0
            merged_provenance = self._merge_unique(self._parse_json_list(existing[2]) if existing else [], incoming_provenance)
            merged_from = self._merge_unique(self._parse_json_list(existing[4]) if existing else [], [pattern.pattern_id])

            near_candidate = conn.execute(
                """
                SELECT pattern_id, canonical_pattern_id, problem_signature, normalized_text
                FROM reasoning_patterns
                WHERE near_fingerprint=?
                  AND pattern_id<>?
                ORDER BY reuse_count DESC, pattern_id ASC
                LIMIT 4
                """,
                (near_fp, pattern.pattern_id),
            )
            near_duplicate_of = None
            for row in near_candidate.fetchall():
                similarity = self._similarity(
                    a_sig=incoming_signature,
                    a_text=incoming_text,
                    b_sig=row[2] or "",
                    b_text=row[3] or "",
                )
                # Safe near-duplicate threshold: link only when very close.
                if similarity >= 0.93:
                    near_duplicate_of = row[1] or row[0]
                    break

            canonical_pattern_id = near_duplicate_of or pattern.pattern_id
            conn.execute(
                """
                INSERT INTO reasoning_patterns(
                    pattern_id, problem_signature, reasoning_steps_json, solution_summary, confidence, provenance_json,
                    memory_subtype, normalized_text, freshness_state, exact_fingerprint, near_fingerprint,
                    canonical_pattern_id, near_duplicate_of, reuse_count, merged_from_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    problem_signature=excluded.problem_signature,
                    reasoning_steps_json=excluded.reasoning_steps_json,
                    solution_summary=excluded.solution_summary,
                    confidence=excluded.confidence,
                    provenance_json=excluded.provenance_json,
                    memory_subtype=excluded.memory_subtype,
                    normalized_text=excluded.normalized_text,
                    freshness_state=excluded.freshness_state,
                    exact_fingerprint=excluded.exact_fingerprint,
                    near_fingerprint=excluded.near_fingerprint,
                    canonical_pattern_id=excluded.canonical_pattern_id,
                    near_duplicate_of=excluded.near_duplicate_of,
                    reuse_count=excluded.reuse_count,
                    merged_from_json=excluded.merged_from_json
                """,
                (
                    pattern.pattern_id,
                    incoming_signature,
                    str(pattern.reasoning_steps),
                    pattern.solution_summary,
                    max(float(existing[1] or 0.0), float(pattern.confidence or 0.0)) if existing else pattern.confidence,
                    json.dumps(merged_provenance),
                    incoming_subtype,
                    incoming_text,
                    getattr(pattern, "freshness_state", None),
                    exact_fp,
                    near_fp,
                    canonical_pattern_id,
                    near_duplicate_of,
                    max(1, existing_reuse + 1),
                    json.dumps(merged_from),
                ),
            )

    def latest_chunk(self) -> dict:
        """Return the most recent reasoning pattern as a RetrievedChunk-compatible dict."""
        with self._connect() as conn:
            try:
                row = conn.execute(
                    "SELECT pattern_id, solution_summary, provenance_json, memory_subtype, normalized_text, freshness_state, "
                    "reuse_count, canonical_pattern_id, near_duplicate_of "
                    "FROM reasoning_patterns ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
            except sqlite3.OperationalError:
                row = conn.execute("SELECT pattern_id, solution_summary FROM reasoning_patterns ORDER BY rowid DESC LIMIT 1").fetchone()
                if not row:
                    return {}
                pattern_id, solution_summary = row
                return {
                    "chunk_id": pattern_id,
                    "document_id": None,
                    "text": solution_summary,
                    "source_type": "reasoning",
                    "score": 0.0,
                    "metadata": {},
                }
            if not row:
                return {}
            pattern_id, solution_summary, provenance_json, memory_subtype, normalized_text, freshness_state, reuse_count, canonical_pattern_id, near_duplicate_of = row
            meta = {}
            try:
                meta['provenance'] = json.loads(provenance_json) if provenance_json else []
            except Exception:
                meta['provenance'] = []
            if memory_subtype:
                meta['memory_subtype'] = memory_subtype
            if normalized_text:
                meta['normalized_text'] = normalized_text
            if freshness_state:
                meta['freshness_state'] = freshness_state
            meta['reuse_count'] = int(reuse_count or 1)
            if canonical_pattern_id:
                meta['canonical_pattern_id'] = canonical_pattern_id
            if near_duplicate_of:
                meta['near_duplicate_of'] = near_duplicate_of
            return {
                "chunk_id": pattern_id,
                "document_id": None,
                "text": solution_summary,
                "source_type": "reasoning",
                "score": 0.0,
                "metadata": meta,
            }

    def query(self, query: str, top_k: int = 5) -> list:
        """Simple query over reasoning patterns: score by token overlap against signature and solution_summary."""
        qtokens = set(query.lower().split())
        results: list[tuple[float, dict]] = []
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT pattern_id, problem_signature, solution_summary, provenance_json, "
                    "memory_subtype, normalized_text, freshness_state, reuse_count, canonical_pattern_id, near_duplicate_of "
                    "FROM reasoning_patterns"
                ).fetchall()
                extracted = []
                for pattern_id, problem_signature, solution_summary, provenance_json, memory_subtype, normalized_text, freshness_state, reuse_count, canonical_pattern_id, near_duplicate_of in rows:
                    extracted.append(
                        (
                            pattern_id,
                            problem_signature,
                            solution_summary,
                            provenance_json,
                            memory_subtype,
                            normalized_text,
                            freshness_state,
                            reuse_count,
                            canonical_pattern_id,
                            near_duplicate_of,
                        )
                    )
            except sqlite3.OperationalError:
                # Older DB schema without provenance_json; fall back gracefully
                rows = conn.execute("SELECT pattern_id, problem_signature, solution_summary FROM reasoning_patterns").fetchall()
                extracted = [
                    (pattern_id, problem_signature, solution_summary, None, None, None, None, 1, None, None)
                    for (pattern_id, problem_signature, solution_summary) in rows
                ]

            for (
                pattern_id,
                problem_signature,
                solution_summary,
                provenance_json,
                memory_subtype,
                normalized_text,
                freshness_state,
                reuse_count,
                canonical_pattern_id,
                near_duplicate_of,
            ) in extracted:
                lexical_base = " ".join(
                    [
                        str(problem_signature or ""),
                        str(solution_summary or ""),
                        str(normalized_text or ""),
                        str(memory_subtype or ""),
                    ]
                )
                tokens = set(lexical_base.lower().split())
                score = len(qtokens & tokens)
                if score > 0:
                    meta = {}
                    try:
                        meta['provenance'] = json.loads(provenance_json) if provenance_json else []
                    except Exception:
                        meta['provenance'] = []
                    if memory_subtype:
                        meta['memory_subtype'] = memory_subtype
                    if normalized_text:
                        meta['normalized_text'] = normalized_text
                    if freshness_state:
                        meta['freshness_state'] = freshness_state
                    meta['reuse_count'] = int(reuse_count or 1)
                    if canonical_pattern_id:
                        meta['canonical_pattern_id'] = canonical_pattern_id
                    if near_duplicate_of:
                        meta['near_duplicate_of'] = near_duplicate_of
                    results.append((float(score), {
                        "chunk_id": pattern_id,
                        "document_id": None,
                        "text": solution_summary,
                        "source_type": "reasoning",
                        "score": float(score),
                        "metadata": meta,
                    }))
        results.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
        out = [r[1] for r in results[:top_k]]
        if not out:
            lc = self.latest_chunk()
            if lc:
                out = [lc]
        return out
