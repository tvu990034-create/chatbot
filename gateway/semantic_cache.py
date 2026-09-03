"""
Semantic Cache
Uses embeddings to cache similar queries, not just exact matches
"""

import hashlib
import json
import os
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np


class SemanticCache:
    """
    Semantic cache that uses embeddings to find similar queries.
    
    Instead of exact string matching, uses cosine similarity on embeddings
    to find semantically similar queries and return cached responses.
    """
    
    def __init__(self, cache_dir: str = "cache", similarity_threshold: float = 0.85):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "semantic_cache.json"
        self.similarity_threshold = similarity_threshold
        self.cache = self._load_cache()
        
        # Try to load embedding model
        self.embedding_model = self._load_embedding_model()
    
    def _load_cache(self) -> Dict:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except:
            pass
    
    def _load_embedding_model(self):
        """Load embedding model for semantic similarity."""
        # Temporarily disabled to avoid slow loading
        # Can be enabled later when sentence-transformers is installed
        return None
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if self.embedding_model is None:
            return None
        
        try:
            embedding = self.embedding_model.encode(text, show_progress_bar=False)
            return embedding
        except Exception:
            return None
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def _exact_key(self, query: str) -> str:
        """Generate exact match key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if available (exact or semantic match).
        
        Args:
            query: The query string
            
        Returns:
            Cached response if found, None otherwise
        """
        # First try exact match
        exact_key = self._exact_key(query)
        if exact_key in self.cache:
            return self.cache[exact_key]['response']
        
        # If embedding model available, try semantic match
        if self.embedding_model is not None:
            query_embedding = self._get_embedding(query)
            if query_embedding is not None:
                # Check all cached queries for similarity
                for key, entry in self.cache.items():
                    if 'embedding' in entry:
                        cached_embedding = np.array(entry['embedding'])
                        similarity = self._cosine_similarity(query_embedding, cached_embedding)
                        
                        if similarity >= self.similarity_threshold:
                            return entry['response']
        
        return None
    
    def set(self, query: str, response: str):
        """
        Cache a response with its embedding.
        
        Args:
            query: The query string
            response: The response to cache
        """
        key = self._exact_key(query)
        
        entry = {
            'query': query,
            'response': response,
            'embedding': None
        }
        
        # Store embedding if model available
        if self.embedding_model is not None:
            embedding = self._get_embedding(query)
            if embedding is not None:
                entry['embedding'] = embedding.tolist()
        
        self.cache[key] = entry
        self._save_cache()
    
    def clear(self):
        """Clear all cached responses."""
        self.cache = {}
        self._save_cache()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "similarity_threshold": self.similarity_threshold,
            "embedding_model_available": self.embedding_model is not None
        }


def get_semantic_cache() -> SemanticCache:
    """Get semantic cache instance."""
    return SemanticCache()
