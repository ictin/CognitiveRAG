import sqlite3
from pathlib import Path
from typing import Optional

class SummaryEdgeStore:
    """Edges connecting summary nodes (from_id, to_id, relation, weight)."""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or 'data/session_memory/summary_edges.sqlite3')
        self._init_db()

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS summary_edges (
                from_id TEXT,
                to_id TEXT,
                relation TEXT,
                weight REAL,
                PRIMARY KEY(from_id, to_id, relation)
            )
            ''')

    def add_edge(self, from_id: str, to_id: str, relation: str, weight: float = 1.0):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO summary_edges(from_id, to_id, relation, weight) VALUES (?, ?, ?, ?) ON CONFLICT(from_id, to_id, relation) DO UPDATE SET weight=excluded.weight',
                (from_id, to_id, relation, weight)
            )

    def get_edges_from(self, from_id: str):
        with self._connect() as conn:
            rows = conn.execute('SELECT to_id, relation, weight FROM summary_edges WHERE from_id=?', (from_id,)).fetchall()
            return [{'to_id': r[0], 'relation': r[1], 'weight': r[2]} for r in rows]
