"""
Enhanced RAG System with Improved Retrieval Quality
Implements advanced retrieval techniques including hybrid search, query expansion, and reranking.
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .retrieval import RetrievalPipeline, RetrievalConfig
from .web_search_tool import WebSearchTool, HybridSearch


@dataclass
class QueryExpansion:
    """Query expansion techniques to improve retrieval."""
    
    @staticmethod
    def expand_query(query: str, method: str = "synonym") -> List[str]:
        """
        Expand query using various techniques.
        
        Args:
            query: Original query
            method: Expansion method (synonym, related, combined)
            
        Returns:
            List of expanded queries
        """
        expanded_queries = [query]
        
        if method == "synonym":
            expanded_queries.extend(QueryExpansion._expand_with_synonyms(query))
        elif method == "related":
            expanded_queries.extend(QueryExpansion._expand_with_related_terms(query))
        elif method == "combined":
            expanded_queries.extend(QueryExpansion._expand_with_synonyms(query))
            expanded_queries.extend(QueryExpansion._expand_with_related_terms(query))
        
        return list(set(expanded_queries))  # Remove duplicates
    
    @staticmethod
    def _expand_with_synonyms(query: str) -> List[str]:
        """Expand query with common synonyms."""
        synonyms = {
            'ai': ['artificial intelligence', 'machine intelligence', 'computational intelligence'],
            'ml': ['machine learning', 'automated learning'],
            'data': ['information', 'dataset', 'statistics'],
            'code': ['programming', 'software', 'development'],
            'app': ['application', 'software', 'program'],
            'web': ['internet', 'online', 'website'],
            'cloud': ['cloud computing', 'distributed computing'],
            'api': ['interface', 'application programming interface'],
            'db': ['database', 'data storage'],
            'ui': ['user interface', 'interface'],
            'ux': ['user experience', 'experience design'],
        }
        
        expanded = []
        words = query.lower().split()
        
        for word in words:
            if word in synonyms:
                for synonym in synonyms[word]:
                    expanded.append(query.lower().replace(word, synonym))
        
        return expanded
    
    @staticmethod
    def _expand_with_related_terms(query: str) -> List[str]:
        """Expand query with related technical terms."""
        related_terms = {
            'python': ['django', 'flask', 'pandas', 'numpy'],
            'javascript': ['react', 'node', 'angular', 'vue'],
            'ai': ['neural network', 'deep learning', 'nlp', 'computer vision'],
            'database': ['sql', 'nosql', 'mysql', 'mongodb', 'postgresql'],
            'security': ['encryption', 'authentication', 'cybersecurity', 'firewall'],
            'performance': ['optimization', 'speed', 'efficiency', 'latency'],
        }
        
        expanded = []
        words = query.lower().split()
        
        for word in words:
            if word in related_terms:
                for term in related_terms[word]:
                    expanded.append(f"{query} {term}")
        
        return expanded


@dataclass
class EnhancedRetrievalConfig(RetrievalConfig):
    """Enhanced configuration with additional retrieval parameters."""
    use_query_expansion: bool = True
    query_expansion_method: str = "combined"
    use_hybrid_search: bool = True
    web_search_weight: float = 0.3
    local_search_weight: float = 0.7
    min_relevance_score: float = 0.3
    max_results_per_source: int = 5


class EnhancedRetrievalPipeline:
    """
    Enhanced RAG pipeline with improved retrieval quality.
    Combines multiple retrieval strategies for better results.
    """
    
    def __init__(
        self,
        documents: List[str],
        doc_ids: List[str],
        cfg: EnhancedRetrievalConfig,
        web_search_tool: Optional[WebSearchTool] = None
    ):
        self.cfg = cfg
        self.documents = documents
        self.doc_ids = doc_ids
        
        # Initialize base retrieval pipeline
        self.base_pipeline = RetrievalPipeline(documents, doc_ids, cfg)
        
        # Initialize web search
        self.web_search = web_search_tool or WebSearchTool()
        self.hybrid_search = HybridSearch(self.base_pipeline, self.web_search)
        
        # Query expansion
        self.query_expansion = QueryExpansion()
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Enhanced retrieval with multiple strategies.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        all_results = []
        
        # Step 1: Query expansion
        expanded_queries = [query]
        if self.cfg.use_query_expansion:
            expanded_queries = self.query_expansion.expand_query(
                query, 
                method=self.cfg.query_expansion_method
            )
        
        # Step 2: Retrieve from multiple sources
        for expanded_query in expanded_queries[:3]:  # Limit to top 3 expansions
            # Local retrieval
            local_results = self.base_pipeline.retrieve(expanded_query)
            for doc_id, content, score in local_results:
                if score >= self.cfg.min_relevance_score:
                    all_results.append({
                        'content': content,
                        'doc_id': doc_id,
                        'score': score * self.cfg.local_search_weight,
                        'source': 'local',
                        'query': expanded_query
                    })
        
        # Step 3: Web search if enabled
        if self.cfg.use_hybrid_search:
            web_results = self.web_search.search(query, self.cfg.max_results_per_source)
            for result in web_results:
                all_results.append({
                    'content': f"{result.title}: {result.snippet}",
                    'doc_id': result.url,
                    'score': self.cfg.web_search_weight,
                    'source': 'web',
                    'url': result.url,
                    'query': query
                })
        
        # Step 4: Deduplicate and rerank
        deduplicated = self._deduplicate_results(all_results)
        reranked = self._rerank_results(deduplicated, query)
        
        # Step 5: Return top-k results
        return reranked[:top_k]
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity."""
        seen = set()
        deduplicated = []
        
        for result in results:
            # Create a simple hash for deduplication
            content_hash = hash(result['content'][:100])  # Use first 100 chars
            if content_hash not in seen:
                seen.add(content_hash)
                deduplicated.append(result)
        
        return deduplicated
    
    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Rerank results based on multiple factors.
        
        Args:
            results: List of retrieved results
            query: Original query for relevance scoring
            
        Returns:
            Reranked list of results
        """
        for result in results:
            # Boost score for exact matches
            if query.lower() in result['content'].lower():
                result['score'] *= 1.2
            
            # Boost score for local sources (more reliable)
            if result['source'] == 'local':
                result['score'] *= 1.1
            
            # Boost score for longer content (more informative)
            content_length = len(result['content'])
            if content_length > 200:
                result['score'] *= 1.05
            elif content_length < 50:
                result['score'] *= 0.9
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def get_context_for_generation(
        self, 
        query: str, 
        max_local: int = 3, 
        max_web: int = 2
    ) -> str:
        """
        Get formatted context for LLM generation.
        
        Args:
            query: Search query
            max_local: Maximum local results
            max_web: Maximum web results
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=max_local + max_web)
        
        context_parts = []
        local_count = 0
        web_count = 0
        
        for result in results:
            if result['source'] == 'local' and local_count < max_local:
                context_parts.append(
                    f"[Knowledge Base - {result['doc_id']}]\n"
                    f"{result['content']}\n"
                )
                local_count += 1
            elif result['source'] == 'web' and web_count < max_web:
                context_parts.append(
                    f"[Web Source]\n"
                    f"{result['content']}\n"
                    f"Source: {result.get('url', 'N/A')}\n"
                )
                web_count += 1
        
        return "\n---\n".join(context_parts)
    
    def add_document(self, doc_id: str, text: str):
        """Add a new document to the retrieval pipeline."""
        self.base_pipeline.add_document(doc_id, text)
        self.documents.append(text)
        self.doc_ids.append(doc_id)
