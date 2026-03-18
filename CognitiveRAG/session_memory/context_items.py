import sqlite3
from pathlib import Path
from typing import Optional

class ContextItemStore:
    """Generic context items referenced by summary nodes or conversation segments."""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or 'data/session_memory/context_items.sqlite3')
        self._init_db()

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS context_items (
                item_id TEXT PRIMARY KEY,
                session_id TEXT,
                type TEXT,
                payload_json TEXT,
                created_at TEXT
            )
            ''')

    def upsert_item(self, item_id: str, session_id: str, type: str, payload_json: str, created_at: str):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO context_items(item_id, session_id, type, payload_json, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(item_id) DO UPDATE SET payload_json=excluded.payload_json, created_at=excluded.created_at',
                (item_id, session_id, type, payload_json, created_at)
            )

    def get_item(self, item_id: str):
        with self._connect() as conn:
            row = conn.execute('SELECT item_id, session_id, type, payload_json, created_at FROM context_items WHERE item_id=?', (item_id,)).fetchone()
            if not row:
                return None
            return {'item_id': row[0], 'session_id': row[1], 'type': row[2], 'payload_json': row[3], 'created_at': row[4]}
