import sqlite3
from pathlib import Path
from typing import Optional

class SummaryNodeStore:
    """Nodes for incremental summaries (small ID, text, created_at, metadata_json)."""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or 'data/session_memory/summary_nodes.sqlite3')
        self._init_db()

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS summary_nodes (
                node_id TEXT PRIMARY KEY,
                session_id TEXT,
                text TEXT,
                metadata_json TEXT,
                created_at TEXT
            )
            ''')

    def upsert_node(self, node_id: str, session_id: str, text: str, metadata_json: str, created_at: str):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO summary_nodes(node_id, session_id, text, metadata_json, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(node_id) DO UPDATE SET text=excluded.text, metadata_json=excluded.metadata_json, created_at=excluded.created_at',
                (node_id, session_id, text, metadata_json, created_at)
            )

    def get_node(self, node_id: str):
        with self._connect() as conn:
            row = conn.execute('SELECT node_id, session_id, text, metadata_json, created_at FROM summary_nodes WHERE node_id=?', (node_id,)).fetchone()
            if not row:
                return None
            return {'node_id': row[0], 'session_id': row[1], 'text': row[2], 'metadata_json': row[3], 'created_at': row[4]}
