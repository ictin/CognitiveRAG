import sqlite3
from pathlib import Path
from typing import Optional

class LargeFileStore:
    """Store metadata for large files and pointers to actual file content on disk.
    Small schema: record_id, file_path, metadata_json, created_at
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or 'data/session_memory/large_files.sqlite3')
        self._init_db()

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS large_files (
                record_id TEXT PRIMARY KEY,
                file_path TEXT,
                metadata_json TEXT,
                created_at TEXT
            )
            ''')

    def upsert_file(self, record_id: str, file_path: str, metadata_json: str, created_at: str):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO large_files(record_id, file_path, metadata_json, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(record_id) DO UPDATE SET file_path=excluded.file_path, metadata_json=excluded.metadata_json, created_at=excluded.created_at',
                (record_id, file_path, metadata_json, created_at)
            )

    def get_file(self, record_id: str):
        with self._connect() as conn:
            row = conn.execute('SELECT record_id, file_path, metadata_json, created_at FROM large_files WHERE record_id=?', (record_id,)).fetchone()
            if not row:
                return None
            return {'record_id': row[0], 'file_path': row[1], 'metadata_json': row[2], 'created_at': row[3]}
