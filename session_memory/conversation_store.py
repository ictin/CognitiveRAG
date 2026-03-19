import sqlite3
from pathlib import Path
from typing import Optional

class ConversationStore:
    """Simple per-session conversation store (lossless append-only rows).
    Minimal schema: conversations table with session_id, message_id, sender, text, created_at
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or 'data/session_memory/conversations.sqlite3')
        self._init_db()

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                sender TEXT,
                text TEXT,
                created_at TEXT,
                PRIMARY KEY(session_id, message_id)
            )
            ''')

    def append_message(self, session_id: str, message_id: str, sender: str, text: str, created_at: str = None):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO conversations(session_id, message_id, sender, text, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(session_id, message_id) DO UPDATE SET sender=excluded.sender, text=excluded.text, created_at=excluded.created_at',
                (session_id, message_id, sender, text, created_at)
            )
        return True

    def upsert_message(self, session_id: str, message_id: str, sender: str, text: str, created_at: str = None):
        return self.append_message(session_id, message_id, sender, text, created_at)

    def add_message(self, session_id: str, message: dict):
        return self.append_message(
            session_id,
            message.get('message_id'),
            message.get('sender'),
            message.get('text'),
            message.get('created_at'),
        )

    def get_messages(self, session_id: str):
        with self._connect() as conn:
            rows = conn.execute('SELECT message_id, sender, text, created_at FROM conversations WHERE session_id=? ORDER BY created_at', (session_id,)).fetchall()
            return [{'message_id': r[0], 'sender': r[1], 'text': r[2], 'created_at': r[3]} for r in rows]

    def list_messages(self, session_id: str):
        return self.get_messages(session_id)
