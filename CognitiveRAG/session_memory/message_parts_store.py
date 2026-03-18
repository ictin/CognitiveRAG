import sqlite3
from pathlib import Path
from typing import Optional

class MessagePartsStore:
    """Store for message parts (tokens, embeddings placeholders, offsets)."""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or 'data/session_memory/message_parts.sqlite3')
        self._init_db()

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS message_parts (
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                part_index INTEGER NOT NULL,
                text TEXT,
                meta_json TEXT,
                PRIMARY KEY(session_id, message_id, part_index)
            )
            ''')

    def add_part(self, session_id: str, message_id: str, part_index: int, text: str, meta_json: str = None):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO message_parts(session_id, message_id, part_index, text, meta_json) VALUES (?, ?, ?, ?, ?)',
                (session_id, message_id, part_index, text, meta_json)
            )

    def get_parts(self, session_id: str, message_id: str):
        with self._connect() as conn:
            rows = conn.execute('SELECT part_index, text, meta_json FROM message_parts WHERE session_id=? AND message_id=? ORDER BY part_index', (session_id, message_id)).fetchall()
            return [{'part_index': r[0], 'text': r[1], 'meta_json': r[2]} for r in rows]
