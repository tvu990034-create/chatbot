"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: Optional[List[Dict]] = None  # RAG source citations
    rating: Optional[str] = None  # User feedback: "up", "down", or None


@dataclass
class SessionInfo:
    """Metadata about a conversation session."""
    session_id: str
    title: str
    created_at: str
    last_activity: str
    message_count: int
    system_prompt: Optional[str] = None


class ConversationStore:
    """
    Async SQLite-based persistent storage for conversation history.
    Uses in-memory cache for fast access and async aiosqlite for non-blocking I/O.
    """
    
    def __init__(self, db_path: str = "data/chatbot.db", max_history_turns: int = 20):
        self.db_path = db_path
        self.max_history_turns = max_history_turns
        # In-memory cache for session history (session_id -> list of messages)
        self._memory_cache: Dict[str, List[ConversationMessage]] = {}
        self._session_metadata: Dict[str, SessionInfo] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def _ensure_db_initialized(self):
        """Initialize SQLite database with required tables (async)."""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        title TEXT,
                        created_at TEXT,
                        last_activity TEXT,
                        message_count INTEGER DEFAULT 0,
                        system_prompt TEXT
                    )
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TEXT,
                        sources TEXT,
                        rating TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    )
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        message_index INTEGER,
                        rating TEXT,
                        timestamp TEXT
                    )
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_session 
                    ON messages(session_id)
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                    ON messages(timestamp)
                """)
                
                await conn.commit()
            
            self._initialized = True
    
    async def create_session(self, session_id: str, system_prompt: Optional[str] = None) -> SessionInfo:
        """Create a new conversation session (async)."""
        await self._ensure_db_initialized()
        now = datetime.utcnow().isoformat()
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, title, created_at, last_activity, message_count, system_prompt)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, "New Chat", now, now, 0, system_prompt))
            await conn.commit()
        
        session = SessionInfo(
            session_id=session_id,
            title="New Chat",
            created_at=now,
            last_activity=now,
            message_count=0,
            system_prompt=system_prompt
        )
        
        # Cache in memory
        self._session_metadata[session_id] = session
        self._memory_cache[session_id] = []
        
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Retrieve session metadata from memory cache (fast path)."""
        return self._session_metadata.get(session_id)
    
    async def update_session_title(self, session_id: str, title: str):
        """Update the title of a session (async, background)."""
        # Update memory cache immediately
        if session_id in self._session_metadata:
            self._session_metadata[session_id].title = title
        
        # Background DB write
        async def _write():
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    UPDATE sessions SET title = ? WHERE session_id = ?
                """, (title, session_id))
                await conn.commit()
        
        asyncio.create_task(_write())
    
    async def update_session_system_prompt(self, session_id: str, system_prompt: str):
        """Update the custom system prompt for a session (async, background)."""
        # Update memory cache immediately
        if session_id in self._session_metadata:
            self._session_metadata[session_id].system_prompt = system_prompt
        
        # Background DB write
        async def _write():
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    UPDATE sessions SET system_prompt = ? WHERE session_id = ?
                """, (system_prompt, session_id))
                await conn.commit()
        
        asyncio.create_task(_write())
    
    def add_message_to_cache(self, session_id: str, role: str, content: str, 
                             sources: Optional[List[Dict]] = None):
        """Add message to in-memory cache (fast path, called during streaming)."""
        if session_id not in self._memory_cache:
            self._memory_cache[session_id] = []
        
        now = datetime.utcnow().isoformat()
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=now,
            sources=sources
        )
        
        self._memory_cache[session_id].append(msg)
        
        # Trim in-memory cache
        if len(self._memory_cache[session_id]) > self.max_history_turns * 2:
            self._memory_cache[session_id] = self._memory_cache[session_id][-self.max_history_turns * 2:]
        
        # Update session metadata in memory
        if session_id in self._session_metadata:
            self._session_metadata[session_id].last_activity = now
            self._session_metadata[session_id].message_count += 1
    
    async def add_message_to_db(self, session_id: str, role: str, content: str, 
                                sources: Optional[List[Dict]] = None):
        """Add message to database (async, background, called after streaming)."""
        now = datetime.utcnow().isoformat()
        sources_json = json.dumps(sources) if sources else None
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT INTO messages (session_id, role, content, timestamp, sources)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, now, sources_json))
            
            await conn.execute("""
                UPDATE sessions 
                SET last_activity = ?, message_count = message_count + 1
                WHERE session_id = ?
            """, (now, session_id))
            
            await conn.commit()
            
            # Trim old messages from DB
            await self._trim_history_db(session_id, conn)
    
    async def _trim_history_db(self, session_id: str, conn: aiosqlite.Connection):
        """Delete old messages from DB to stay within max_history_turns limit."""
        cursor = await conn.execute("""
            SELECT COUNT(*) FROM messages WHERE session_id = ?
        """, (session_id,))
        count = (await cursor.fetchone())[0]
        
        if count > self.max_history_turns * 2:  # Keep pairs (user + assistant)
            await conn.execute("""
                DELETE FROM messages 
                WHERE session_id = ? 
                AND id NOT IN (
                    SELECT id FROM messages 
                    WHERE session_id = ? 
                    ORDER BY id DESC 
                    LIMIT ?
                )
            """, (session_id, session_id, self.max_history_turns * 2))
    
    def get_conversation_history(self, session_id: str) -> List[ConversationMessage]:
        """Retrieve conversation history from memory cache (fast path)."""
        return self._memory_cache.get(session_id, [])
    
    async def load_conversation_history(self, session_id: str) -> List[ConversationMessage]:
        """Load conversation history from DB into memory cache (async, one-time load)."""
        # Check if already in cache
        if session_id in self._memory_cache:
            return self._memory_cache[session_id]
        
        await self._ensure_db_initialized()
        
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("""
                SELECT role, content, timestamp, sources, rating
                FROM messages 
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))
            
            messages = []
            async for row in cursor:
                sources = json.loads(row[3]) if row[3] else None
                messages.append(ConversationMessage(
                    role=row[0],
                    content=row[1],
                    timestamp=row[2],
                    sources=sources,
                    rating=row[4]
                ))
        
        # Cache in memory
        self._memory_cache[session_id] = messages
        
        return messages
    
    async def add_feedback(self, session_id: str, message_index: int, rating: str):
        """Record user feedback for a message (async, background)."""
        now = datetime.utcnow().isoformat()
        
        async def _write():
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    INSERT INTO feedback (session_id, message_index, rating, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (session_id, message_index, rating, now))
                
                await conn.execute("""
                    UPDATE messages SET rating = ?
                    WHERE session_id = ? AND id = ?
                """, (rating, session_id, message_index + 1))
                
                await conn.commit()
        
        asyncio.create_task(_write())
    
    async def list_all_sessions(self) -> List[SessionInfo]:
        """List all sessions with metadata (async)."""
        await self._ensure_db_initialized()
        
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("""
                SELECT session_id, title, created_at, last_activity, message_count, system_prompt
                FROM sessions
                ORDER BY last_activity DESC
            """)
            
            sessions = []
            async for row in cursor:
                sessions.append(SessionInfo(
                    session_id=row[0],
                    title=row[1],
                    created_at=row[2],
                    last_activity=row[3],
                    message_count=row[4],
                    system_prompt=row[5]
                ))
        
        return sessions
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages (async)."""
        # Remove from memory cache immediately
        self._memory_cache.pop(session_id, None)
        self._session_metadata.pop(session_id, None)
        
        await self._ensure_db_initialized()
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                DELETE FROM sessions WHERE session_id = ?
            """, (session_id,))
            await conn.commit()
            
            return True
    
    async def get_usage_stats(self) -> Dict:
        """Get usage statistics for admin dashboard (async)."""
        await self._ensure_db_initialized()
        
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = (await cursor.fetchone())[0]
            
            cursor = await conn.execute("SELECT COUNT(*) FROM messages")
            total_messages = (await cursor.fetchone())[0]
            
            cursor = await conn.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = (await cursor.fetchone())[0]
            
            cursor = await conn.execute("""
                SELECT rating, COUNT(*) FROM feedback GROUP BY rating
            """)
            feedback_breakdown = {}
            async for row in cursor:
                feedback_breakdown[row[0]] = row[1]
            
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM sessions 
                WHERE datetime(last_activity) > datetime('now', '-1 day')
            """)
            active_sessions = (await cursor.fetchone())[0]
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_feedback": total_feedback,
            "feedback_breakdown": feedback_breakdown,
            "active_sessions_24h": active_sessions
        }
