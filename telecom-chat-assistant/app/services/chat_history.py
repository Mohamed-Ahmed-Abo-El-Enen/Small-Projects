import sqlite3
import threading
import time
import hashlib
import json
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import settings


class ChatHistoryManager:
    """Manage chat history using SQLite database (Singleton)"""

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.db_path = str(settings.CHAT_HISTORY_DB)
        self._init_database()
        self._initialized = True
        print(f"ChatHistoryManager initialized (Singleton) at {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                language TEXT,
                sources TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_id
            ON messages(conversation_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON messages(timestamp)
        """)

        conn.commit()
        conn.close()
        print(f"Chat history database initialized: {self.db_path}")

    def create_conversation(self, user_id: str = "default", metadata: Dict = None) -> str:
        """Create a new conversation and return its ID"""
        conversation_id = f"{user_id}_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO conversations (conversation_id, user_id, metadata)
                VALUES (?, ?, ?)
            """, (conversation_id, user_id, json.dumps(metadata or {})))

            conn.commit()
            conn.close()

        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: str,
                    language: str = None, sources: List[Dict] = None):
        """Add a message to conversation history (thread-safe)"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO conversations (conversation_id, user_id)
                VALUES (?, ?)
            """, (conversation_id, "default"))

            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, language, sources)
                VALUES (?, ?, ?, ?, ?)
            """, (conversation_id, role, content, language, json.dumps(sources or [])))

            cursor.execute("""
                UPDATE conversations
                SET last_updated = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
            """, (conversation_id,))

            conn.commit()
            conn.close()

    def get_conversation_history(self, conversation_id: str,
                                 limit: int = None) -> List[Dict]:
        """Get conversation history"""
        limit = limit or settings.MAX_HISTORY_MESSAGES

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT role, content, timestamp, language, sources
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (conversation_id, limit))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row[0],
                'content': row[1],
                'timestamp': row[2],
                'language': row[3],
                'sources': json.loads(row[4]) if row[4] else []
            })

        conn.close()
        return list(reversed(messages))

    def get_formatted_history(self, conversation_id: str,
                              limit: int = None) -> List[HumanMessage | AIMessage]:
        """Get conversation history formatted as LangChain messages"""
        history = self.get_conversation_history(conversation_id, limit)

        messages = []
        for msg in history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))

        return messages

    def get_all_conversations(self, user_id: str = None) -> List[Dict]:
        """Get all conversations for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if user_id:
            cursor.execute("""
                SELECT conversation_id, created_at, last_updated, metadata
                FROM conversations
                WHERE user_id = ?
                ORDER BY last_updated DESC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT conversation_id, created_at, last_updated, metadata
                FROM conversations
                ORDER BY last_updated DESC
            """)

        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'conversation_id': row[0],
                'created_at': row[1],
                'last_updated': row[2],
                'metadata': json.loads(row[3]) if row[3] else {}
            })

        conn.close()
        return conversations

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and its messages"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))

            conn.commit()
            conn.close()

    def get_conversation_summary(self, conversation_id: str) -> Dict:
        """Get summary statistics for a conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM messages
            WHERE conversation_id = ?
        """, (conversation_id,))

        row = cursor.fetchone()
        conn.close()

        return {
            'total_messages': row[0],
            'first_message': row[1],
            'last_message': row[2]
        }