"""Local SQLite persistence for chat transcripts and retrieved product context."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ChatMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path(__file__).resolve().parent.parent / "data" / "chat_history.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def load_messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT role, content, result_json FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        messages = []
        for role, content, result_json in rows:
            item = {"role": role, "content": content}
            if result_json:
                item["result"] = json.loads(result_json)
            messages.append(item)
        return messages

    def save_message(self, conversation_id: str, message: dict) -> None:
        result_json = json.dumps(message["result"], ensure_ascii=False) if message.get("result") else None
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages (conversation_id, role, content, result_json) VALUES (?, ?, ?, ?)",
                (conversation_id, message["role"], message["content"], result_json),
            )

    def clear(self, conversation_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
