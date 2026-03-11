"""
Local SQLite-based conversation manager for offline development.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import uuid


class LocalConversationManager:
    """Manages conversation history using local SQLite database."""
    
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path(__file__).parent / "conversations.db"
        
        self.db_path = db_path
        self.current_conversation_id = None
        self.current_history = []
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database with tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ===== Users =====
    def create_user(self, email: str, username: str) -> dict:
        """Create a new user."""
        try:
            user_id = str(uuid.uuid4())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO users (id, email, username) VALUES (?, ?, ?)',
                (user_id, email, username)
            )
            conn.commit()
            conn.close()
            
            return {"id": user_id, "email": email, "username": username}
        except sqlite3.IntegrityError:
            return None
    
    def get_user_by_email(self, email: str) -> dict:
        """Get user by email."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, username FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {"id": row[0], "email": row[1], "username": row[2]}
        return None
    
    def get_user_by_id(self, user_id: str) -> dict:
        """Get user by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, username FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {"id": row[0], "email": row[1], "username": row[2]}
        return None
    
    # ===== Conversations =====
    def create_conversation(self, user_id: str) -> str:
        """Create a new conversation for a user."""
        conv_id = str(uuid.uuid4())[:8]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO conversations (id, user_id) VALUES (?, ?)',
            (conv_id, user_id)
        )
        conn.commit()
        conn.close()
        
        return conv_id
    
    def get_conversations(self, user_id: str) -> list:
        """Get all conversations for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [{"id": row[0], "created_at": row[1]} for row in rows]
    
    # ===== Messages =====
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """Add a message to a conversation."""
        try:
            msg_id = str(uuid.uuid4())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)',
                (msg_id, conversation_id, role, content)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding message: {e}")
            return False
    
    def get_messages(self, conversation_id: str) -> list:
        """Get all messages in a conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC',
            (conversation_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in rows]
