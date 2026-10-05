"""
Learning System for Chatbot
Enables the chatbot to learn from user interactions and feedback
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import aiosqlite
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class LearningExample:
    """A learned Q&A pair from user interactions."""
    question: str
    answer: str
    confidence: float
    source: str  # "user_feedback", "conversation", "correction"
    timestamp: str
    usage_count: int = 0
    success_rate: float = 1.0


class LearningSystem:
    """
    Learning system that improves the chatbot over time.
    Learns from user feedback, successful responses, and corrections.
    """
    
    def __init__(self, db_path: str = "data/learning.db"):
        self.db_path = db_path
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._cache: Dict[str, LearningExample] = {}
        
    async def _ensure_db_initialized(self):
        """Initialize the learning database."""
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
                    CREATE TABLE IF NOT EXISTS learning_examples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question_hash TEXT UNIQUE,
                        question TEXT,
                        answer TEXT,
                        confidence REAL,
                        source TEXT,
                        timestamp TEXT,
                        usage_count INTEGER DEFAULT 0,
                        success_count INTEGER DEFAULT 0,
                        total_attempts INTEGER DEFAULT 0
                    )
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_question_hash 
                    ON learning_examples(question_hash)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_confidence 
                    ON learning_examples(confidence)
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        question TEXT,
                        answer TEXT,
                        rating TEXT,
                        timestamp TEXT
                    )
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_rating 
                    ON feedback_log(rating)
                """)
                
                await conn.commit()
            
            # Load cache
            await self._load_cache()
            self._initialized = True
    
    async def _load_cache(self):
        """Load learning examples into cache."""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("""
                SELECT question, answer, confidence, source, timestamp, 
                       usage_count, success_count, total_attempts
                FROM learning_examples
                WHERE confidence > 0.5
            """)
            
            async for row in cursor:
                success_rate = row[6] / row[7] if row[7] > 0 else 1.0
                example = LearningExample(
                    question=row[0],
                    answer=row[1],
                    confidence=row[2],
                    source=row[3],
                    timestamp=row[4],
                    usage_count=row[5],
                    success_rate=success_rate
                )
                question_hash = self._hash_question(row[0])
                self._cache[question_hash] = example
    
    def _hash_question(self, question: str) -> str:
        """Create a hash for question deduplication."""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    async def learn_from_feedback(
        self, 
        session_id: str, 
        question: str, 
        answer: str, 
        rating: str
    ):
        """
        Learn from user feedback on a response.
        rating: "up" (good), "down" (bad), or None
        """
        await self._ensure_db_initialized()
        now = datetime.utcnow().isoformat()
        
        # Log feedback
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT INTO feedback_log (session_id, question, answer, rating, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, question, answer, rating, now))
            await conn.commit()
        
        # If positive feedback, add to learning examples
        if rating == "up":
            await self._add_learning_example(
                question=question,
                answer=answer,
                confidence=0.8,
                source="user_feedback"
            )
        
        # If negative feedback, reduce confidence
        elif rating == "down":
            question_hash = self._hash_question(question)
            if question_hash in self._cache:
                self._cache[question_hash].confidence *= 0.7
                self._cache[question_hash].success_rate *= 0.8
                
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute("""
                        UPDATE learning_examples 
                        SET confidence = confidence * 0.7
                        WHERE question_hash = ?
                    """, (question_hash,))
                    await conn.commit()
    
    async def learn_from_conversation(
        self, 
        conversation_history: List[Dict[str, str]]
    ):
        """
        Learn from successful conversation patterns.
        Extracts Q&A pairs that led to positive outcomes.
        """
        await self._ensure_db_initialized()
        
        # Extract question-answer pairs from conversation
        for i in range(0, len(conversation_history) - 1, 2):
            if i + 1 < len(conversation_history):
                user_msg = conversation_history[i]
                assistant_msg = conversation_history[i + 1]
                
                if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                    question = user_msg.get("content", "")
                    answer = assistant_msg.get("content", "")
                    
                    if question and answer:
                        # Add with moderate confidence
                        await self._add_learning_example(
                            question=question,
                            answer=answer,
                            confidence=0.6,
                            source="conversation"
                        )
    
    async def learn_correction(
        self, 
        original_question: str, 
        original_answer: str, 
        corrected_answer: str
    ):
        """
        Learn from user corrections.
        When user provides a better answer, replace the learned one.
        """
        await self._ensure_db_initialized()
        
        question_hash = self._hash_question(original_question)
        
        # Update existing or add new
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO learning_examples 
                (question_hash, question, answer, confidence, source, timestamp, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT usage_count FROM learning_examples WHERE question_hash = ?), 0))
            """, (
                question_hash, 
                original_question, 
                corrected_answer, 
                0.9,  # High confidence for user corrections
                "correction",
                datetime.utcnow().isoformat(),
                question_hash
            ))
            await conn.commit()
        
        # Update cache
        self._cache[question_hash] = LearningExample(
            question=original_question,
            answer=corrected_answer,
            confidence=0.9,
            source="correction",
            timestamp=datetime.utcnow().isoformat(),
            usage_count=self._cache.get(question_hash, LearningExample("", "", 0, "", "")).usage_count,
            success_rate=1.0
        )
    
    async def _add_learning_example(
        self, 
        question: str, 
        answer: str, 
        confidence: float, 
        source: str
    ):
        """Add a new learning example."""
        question_hash = self._hash_question(question)
        
        # Check if already exists
        if question_hash in self._cache:
            # Update confidence if higher
            if confidence > self._cache[question_hash].confidence:
                self._cache[question_hash].confidence = confidence
                self._cache[question_hash].answer = answer
                
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute("""
                        UPDATE learning_examples 
                        SET confidence = ?, answer = ?
                        WHERE question_hash = ?
                    """, (confidence, answer, question_hash))
                    await conn.commit()
            return
        
        # Add new example
        example = LearningExample(
            question=question,
            answer=answer,
            confidence=confidence,
            source=source,
            timestamp=datetime.utcnow().isoformat()
        )
        self._cache[question_hash] = example
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR IGNORE INTO learning_examples 
                (question_hash, question, answer, confidence, source, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (question_hash, question, answer, confidence, source, example.timestamp))
            await conn.commit()
    
    async def get_learned_answer(self, question: str) -> Optional[str]:
        """
        Get a learned answer for a question if available.
        Returns the best matching learned answer.
        """
        await self._ensure_db_initialized()
        
        question_hash = self._hash_question(question)
        
        # Exact match
        if question_hash in self._cache:
            example = self._cache[question_hash]
            if example.confidence > 0.5:
                # Update usage
                example.usage_count += 1
                await self._update_usage(question_hash)
                return example.answer
        
        # Fuzzy match (find similar questions)
        similar = await self._find_similar_questions(question)
        if similar:
            best_match = max(similar, key=lambda x: x[1])
            if best_match[1] > 0.7:  # High similarity threshold
                question_hash = best_match[0]
                if question_hash in self._cache:
                    example = self._cache[question_hash]
                    example.usage_count += 1
                    await self._update_usage(question_hash)
                    return example.answer
        
        return None
    
    async def _find_similar_questions(self, question: str) -> List[Tuple[str, float]]:
        """Find similar questions using simple keyword matching."""
        question_words = set(question.lower().split())
        similar = []
        
        for q_hash, example in self._cache.items():
            example_words = set(example.question.lower().split())
            intersection = question_words & example_words
            union = question_words | example_words
            
            if union:
                similarity = len(intersection) / len(union)
                if similarity > 0.3:  # Minimum similarity
                    similar.append((q_hash, similarity))
        
        return similar
    
    async def _update_usage(self, question_hash: str):
        """Update usage statistics for a learning example."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                UPDATE learning_examples 
                SET usage_count = usage_count + 1
                WHERE question_hash = ?
            """, (question_hash,))
            await conn.commit()
    
    async def record_usage_result(self, question: str, success: bool):
        """
        Record whether a learned answer was successful.
        Used to improve confidence scores over time.
        """
        await self._ensure_db_initialized()
        question_hash = self._hash_question(question)
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                UPDATE learning_examples 
                SET total_attempts = total_attempts + 1,
                    success_count = success_count + ?
                WHERE question_hash = ?
            """, (1 if success else 0, question_hash))
            await conn.commit()
        
        # Update cache
        if question_hash in self._cache:
            example = self._cache[question_hash]
            example.total_attempts = getattr(example, 'total_attempts', 0) + 1
            example.success_count = getattr(example, 'success_count', 0) + (1 if success else 0)
            example.success_rate = example.success_count / example.total_attempts if example.total_attempts > 0 else 1.0
            
            # Adjust confidence based on success rate
            if example.total_attempts > 5:
                if example.success_rate > 0.8:
                    example.confidence = min(0.95, example.confidence + 0.05)
                elif example.success_rate < 0.5:
                    example.confidence = max(0.1, example.confidence - 0.1)
    
    async def get_learning_stats(self) -> Dict:
        """Get statistics about the learning system."""
        await self._ensure_db_initialized()
        
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM learning_examples")
            total_examples = (await cursor.fetchone())[0]
            
            cursor = await conn.execute("SELECT AVG(confidence) FROM learning_examples")
            avg_confidence = (await cursor.fetchone())[0] or 0.0
            
            cursor = await conn.execute("SELECT SUM(usage_count) FROM learning_examples")
            total_usage = (await cursor.fetchone())[0] or 0
            
            cursor = await conn.execute("SELECT source, COUNT(*) FROM learning_examples GROUP BY source")
            source_breakdown = {}
            async for row in cursor:
                source_breakdown[row[0]] = row[1]
            
            cursor = await conn.execute("SELECT COUNT(*) FROM feedback_log WHERE rating = 'up'")
            positive_feedback = (await cursor.fetchone())[0] or 0
            
            cursor = await conn.execute("SELECT COUNT(*) FROM feedback_log WHERE rating = 'down'")
            negative_feedback = (await cursor.fetchone())[0] or 0
        
        return {
            "total_learned_examples": total_examples,
            "average_confidence": avg_confidence,
            "total_usage": total_usage,
            "source_breakdown": source_breakdown,
            "positive_feedback_count": positive_feedback,
            "negative_feedback_count": negative_feedback
        }
    
    async def export_learned_knowledge(self) -> List[Dict]:
        """
        Export learned knowledge for integration into the main knowledge base.
        Returns high-confidence examples that can be added to documents.
        """
        await self._ensure_db_initialized()
        
        learned_knowledge = []
        
        for q_hash, example in self._cache.items():
            if example.confidence > 0.7 and example.usage_count > 2:
                learned_knowledge.append({
                    "question": example.question,
                    "answer": example.answer,
                    "confidence": example.confidence,
                    "usage_count": example.usage_count,
                    "success_rate": example.success_rate
                })
        
        # Sort by confidence and usage
        learned_knowledge.sort(key=lambda x: (x["confidence"], x["usage_count"]), reverse=True)
        
        return learned_knowledge
    
    async def prune_low_confidence(self, min_confidence: float = 0.3, min_usage: int = 5):
        """
        Remove low-confidence and rarely-used examples.
        Keeps the knowledge base clean and efficient.
        """
        await self._ensure_db_initialized()
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                DELETE FROM learning_examples 
                WHERE confidence < ? AND usage_count < ?
            """, (min_confidence, min_usage))
            await conn.commit()
        
        # Update cache
        to_remove = []
        for q_hash, example in self._cache.items():
            if example.confidence < min_confidence and example.usage_count < min_usage:
                to_remove.append(q_hash)
        
        for q_hash in to_remove:
            del self._cache[q_hash]
        
        logger.info(f"Pruned {len(to_remove)} low-confidence learning examples")
