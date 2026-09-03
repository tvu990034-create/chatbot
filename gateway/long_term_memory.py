"""
Long-term Memory Module
Implements advanced memory mechanisms: Attention, Contrastive Learning, Modern Hopfield, RAG, Experience Replay, Working Memory
Based on neuroscience and modern memory research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from dataclasses import dataclass, field
from collections import deque
import random

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A single memory item."""
    content: Any
    embedding: Optional[np.ndarray] = None
    timestamp: float = 0.0
    access_count: int = 0
    importance: float = 1.0


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention: Retrieval focus mechanism
    Retrieves weighted sum of values based on query-key compatibility
    """
    
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
    
    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                  mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor (batch, heads, seq_len, dim)
            K: Key tensor (batch, heads, seq_len, dim)
            V: Value tensor (batch, heads, seq_len, dim)
            mask: Optional attention mask
        
        Returns:
            Output and attention weights
        """
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = F.softmax(scores, dim=-1)
        output = torch.matmul(p_attn, V)
        
        return output, p_attn
    
    def retrieve_memory(self, query: np.ndarray, memory_keys: np.ndarray, 
                       memory_values: np.ndarray, top_k: int = 5) -> List[Tuple]:
        """
        Retrieve top-k memories based on query.
        
        Args:
            query: Query vector
            memory_keys: Memory key vectors
            memory_values: Memory value vectors
            top_k: Number of memories to retrieve
        
        Returns:
            List of (value, attention_weight) tuples
        """
        Q = torch.tensor(query).unsqueeze(0).unsqueeze(0)
        K = torch.tensor(memory_keys).unsqueeze(0).unsqueeze(0)
        V = torch.tensor(memory_values).unsqueeze(0).unsqueeze(0)
        
        _, p_attn = self.attention(Q, K, V)
        
        # Get top-k indices
        top_k_indices = torch.topk(p_attn.squeeze(), top_k).indices
        
        results = []
        for idx in top_k_indices:
            results.append((memory_values[idx], p_attn[0, 0, idx].item()))
        
        return results


class ContrastiveLearning:
    """
    Contrastive Learning (InfoNCE): Memory embedding shaping
    Pulls positive memories close, pushes negative memories far away
    """
    
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
    
    def info_nce_loss(self, query: torch.Tensor, positive_key: torch.Tensor, 
                      negative_keys: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query vector (batch, dim)
            positive_key: Positive key (batch, dim)
            negative_keys: Negative keys (batch, n_neg, dim)
        
        Returns:
            InfoNCE loss
        """
        batch_size = query.size(0)
        query = F.normalize(query, dim=1)
        positive_key = F.normalize(positive_key, dim=1)
        negative_keys = F.normalize(negative_keys, dim=2)
        
        pos_sim = (query * positive_key).sum(dim=-1) / self.temperature
        neg_sim = torch.bmm(negative_keys, query.unsqueeze(-1)).squeeze(-1) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        
        return F.cross_entropy(logits, labels)
    
    def update_embeddings(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update embeddings using contrastive learning.
        
        Args:
            embeddings: Current embeddings
            labels: Positive/negative labels
        
        Returns:
            Updated embeddings
        """
        # Simplified contrastive update
        query = torch.tensor(embeddings)
        positive_mask = (labels == 1)
        
        # Pull positives closer
        if positive_mask.any():
            positive_mean = query[positive_mask].mean(dim=0, keepdim=True)
            embeddings = embeddings + 0.1 * (positive_mean.numpy() - embeddings)
        
        return embeddings


class ModernHopfieldNetwork:
    """
    Modern Hopfield Network: Exponential capacity associative memory
    Single-step, continuous-state associative memory
    """
    
    def __init__(self, beta: float = 10.0):
        self.beta = beta
        self.memory_matrix = None
    
    def store_patterns(self, patterns: np.ndarray):
        """
        Store patterns in memory.
        
        Args:
            patterns: Memory patterns (num_patterns, dim)
        """
        self.memory_matrix = patterns
    
    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """
        Retrieve memory pattern from query.
        
        Args:
            query: Query vector (dim,)
        
        Returns:
            Retrieved memory vector
        """
        if self.memory_matrix is None:
            return query
        
        query_tensor = torch.tensor(query).unsqueeze(0)
        memory_tensor = torch.tensor(self.memory_matrix)
        
        attention_weights = F.softmax(self.beta * torch.matmul(query_tensor, memory_tensor.T), dim=-1)
        retrieved = torch.matmul(attention_weights, memory_tensor)
        
        return retrieved.squeeze(0).numpy()


class RetrievalAugmentedGeneration:
    """
    RAG: Retrieval-Augmented Generation
    Marginalizes over retrieved documents to ground generation
    """
    
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.document_embeddings = None
        self.document_texts = None
    
    def index_documents(self, documents: List[str], embeddings: np.ndarray):
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings
        """
        self.document_texts = documents
        self.document_embeddings = embeddings
    
    def retrieve(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for query.
        
        Args:
            query_embedding: Query vector
        
        Returns:
            List of (document, score) tuples
        """
        if self.document_embeddings is None:
            return []
        
        query_tensor = torch.tensor(query_embedding)
        doc_tensor = torch.tensor(self.document_embeddings)
        
        scores = torch.matmul(query_tensor, doc_tensor.T)
        top_k_indices = torch.topk(scores, min(self.top_k, len(self.document_texts))).indices
        
        results = []
        for idx in top_k_indices:
            results.append((self.document_texts[idx], scores[idx].item()))
        
        return results
    
    def augment_generation(self, query: str, retrieved_docs: List[str]) -> str:
        """
        Augment query with retrieved context.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents
        
        Returns:
            Augmented query with context
        """
        context = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
        return f"{context}\n\nQuestion: {query}"


class ExperienceReplayBuffer:
    """
    Experience Replay: Buffer for past experiences
    Interleaves current with past experiences to prevent catastrophic forgetting
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
    
    def store(self, transition: Tuple):
        """
        Store a transition in the buffer.
        
        Args:
            transition: (state, action, reward, next_state, done)
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample random transitions from buffer.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Batch of transitions
        """
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class WorkingMemory:
    """
    Working Memory: Gated recurrence for dynamic scratchpad
    Controls whether to overwrite or retain current memory state
    """
    
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.W_gate = np.random.randn(dim, dim) * 0.1
        self.U_gate = np.random.randn(dim, dim) * 0.1
        self.state = np.zeros(dim)
    
    def step(self, x_t: np.ndarray) -> np.ndarray:
        """
        Update working memory state.
        
        Args:
            x_t: New input vector
        
        Returns:
            New memory state
        """
        gate = 1 / (1 + np.exp(-(np.dot(x_t, self.W_gate) + np.dot(self.state, self.U_gate))))
        self.state = (1 - gate) * self.state + gate * x_t
        return self.state
    
    def reset(self):
        """Reset working memory state."""
        self.state = np.zeros(self.dim)


class LongTermMemorySuite:
    """
    Complete suite of long-term memory components
    Integrates attention, contrastive learning, hopfield, RAG, replay, and working memory
    """
    
    def __init__(self, embedding_dim: int = 512, buffer_capacity: int = 10000):
        logger.info("Initializing Long-term Memory Suite")
        
        # Retrieval Mechanisms
        self.attention = ScaledDotProductAttention(d_model=embedding_dim)
        self.contrastive = ContrastiveLearning()
        self.hopfield = ModernHopfieldNetwork()
        self.rag = RetrievalAugmentedGeneration()
        
        # Memory Systems
        self.replay_buffer = ExperienceReplayBuffer(capacity=buffer_capacity)
        self.working_memory = WorkingMemory(dim=embedding_dim)
        
        # Memory Storage
        self.episodic_memory = []  # List of MemoryItem
        self.semantic_memory = {}  # Key-value store
        
        logger.info("Long-term Memory Suite initialized")
    
    def store_episodic(self, content: Any, embedding: Optional[np.ndarray] = None, 
                       importance: float = 1.0):
        """
        Store episodic memory.
        
        Args:
            content: Memory content
            embedding: Content embedding
            importance: Memory importance weight
        """
        item = MemoryItem(
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            importance=importance
        )
        self.episodic_memory.append(item)
        
        # Also store in hopfield if embedding provided
        if embedding is not None:
            if self.hopfield.memory_matrix is None:
                self.hopfield.store_patterns(np.array([embedding]))
            else:
                patterns = np.vstack([self.hopfield.memory_matrix, embedding])
                self.hopfield.store_patterns(patterns)
    
    def retrieve_episodic(self, query: np.ndarray, top_k: int = 5) -> List[Any]:
        """
        Retrieve episodic memories.
        
        Args:
            query: Query vector
            top_k: Number of memories to retrieve
        
        Returns:
            List of retrieved content
        """
        if not self.episodic_memory:
            return []
        
        # Get embeddings
        embeddings = np.array([item.embedding for item in self.episodic_memory if item.embedding is not None])
        if len(embeddings) == 0:
            return []
        
        values = np.array([item.content for item in self.episodic_memory if item.embedding is not None])
        
        # Use attention to retrieve
        results = self.attention.retrieve_memory(query, embeddings, values, top_k)
        return [content for content, _ in results]
    
    def store_semantic(self, key: str, value: Any):
        """
        Store semantic memory (key-value).
        
        Args:
            key: Memory key
            value: Memory value
        """
        self.semantic_memory[key] = value
    
    def retrieve_semantic(self, key: str) -> Optional[Any]:
        """
        Retrieve semantic memory.
        
        Args:
            key: Memory key
        
        Returns:
            Memory value or None
        """
        return self.semantic_memory.get(key)
    
    def augment_with_rag(self, query: str, query_embedding: np.ndarray) -> str:
        """
        Augment query with RAG retrieval.
        
        Args:
            query: Original query
            query_embedding: Query embedding
        
        Returns:
            Augmented query
        """
        if self.rag.document_embeddings is None:
            return query
        
        retrieved = self.rag.retrieve(query_embedding)
        return self.rag.augment_generation(query, [doc for doc, _ in retrieved])
    
    def update_working_memory(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Update working memory with new input.
        
        Args:
            input_vector: New input vector
        
        Returns:
            Updated working memory state
        """
        return self.working_memory.step(input_vector)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for monitoring."""
        return {
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "replay_buffer_size": self.replay_buffer.size(),
            "working_memory_norm": np.linalg.norm(self.working_memory.state),
            "hopfield_patterns": self.hopfield.memory_matrix.shape[0] if self.hopfield.memory_matrix is not None else 0
        }