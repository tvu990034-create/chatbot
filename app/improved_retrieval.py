"""
Improved Retrieval System with Hybrid Search
Addresses the core retrieval quality issues in the original system
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ImprovedRetrievalConfig:
    """Configuration for improved retrieval"""
    use_semantic_search: bool = True
    use_keyword_search: bool = True
    semantic_weight: float = 0.7  # Weight for semantic search
    keyword_weight: float = 0.3   # Weight for keyword search
    top_k: int = 5
    min_score_threshold: float = 0.3
    enable_query_expansion: bool = True

class ImprovedRetriever:
    """Hybrid retrieval with semantic + keyword search"""
    
    def __init__(self, documents: List[str], doc_ids: List[str], config: ImprovedRetrievalConfig = None):
        self.documents = documents
        self.doc_ids = doc_ids
        self.config = config or ImprovedRetrievalConfig()
        self._build_index()
        
    def _build_index(self):
        """Build retrieval indices"""
        logger.info(f"Building retrieval index for {len(self.documents)} documents")
        
        # Build keyword index (simple TF-IDF style)
        self.keyword_index = self._build_keyword_index()
        
        # Build semantic index if available
        if self.config.use_semantic_search:
            self.semantic_index = self._build_semantic_index()
        else:
            self.semantic_index = None
    
    def _build_keyword_index(self):
        """Build simple keyword index"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(self.documents)
            return {
                'vectorizer': vectorizer,
                'matrix': tfidf_matrix
            }
        except ImportError:
            logger.warning("sklearn not available, using fallback keyword search")
            return None
    
    def _build_semantic_index(self):
        """Build semantic search index"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(self.documents, show_progress_bar=True)
            return {
                'model': model,
                'embeddings': embeddings
            }
        except ImportError:
            logger.warning("sentence-transformers not available, semantic search disabled")
            return None
    
    def retrieve(self, query: str) -> List[Tuple[str, str, float]]:
        """Hybrid retrieval combining semantic and keyword search"""
        results = []
        
        # Keyword search
        if self.config.use_keyword_search and self.keyword_index:
            keyword_results = self._keyword_search(query)
            results.extend(keyword_results)
        
        # Semantic search
        if self.config.use_semantic_search and self.semantic_index:
            semantic_results = self._semantic_search(query)
            results.extend(semantic_results)
        
        # Combine and rerank results
        combined_results = self._combine_results(results)
        
        # Filter by threshold
        filtered_results = [(doc_id, doc, score) for doc_id, doc, score in combined_results 
                             if score >= self.config.min_score_threshold]
        
        # Return top-k
        return sorted(filtered_results, key=lambda x: x[2], reverse=True)[:self.config.top_k]
    
    def _keyword_search(self, query: str) -> List[Tuple[str, str, float]]:
        """Keyword-based search using TF-IDF"""
        if not self.keyword_index:
            return []
        
        vectorizer = self.keyword_index['vectorizer']
        matrix = self.keyword_index['matrix']
        
        query_vec = vectorizer.transform([query])
        similarities = (matrix * query_vec.T).toarray().flatten()
        
        results = []
        for idx, score in enumerate(similarities):
            if score > 0:
                results.append((self.doc_ids[idx], self.documents[idx], float(score)))
        
        return results
    
    def _semantic_search(self, query: str) -> List[Tuple[str, str, float]]:
        """Semantic search using embeddings"""
        if not self.semantic_index:
            return []
        
        model = self.semantic_index['model']
        embeddings = self.semantic_index['embeddings']
        
        query_embedding = model.encode([query])
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        results = []
        for idx, score in enumerate(similarities):
            if score > 0:
                results.append((self.doc_ids[idx], self.documents[idx], float(score)))
        
        return results
    
    def _combine_results(self, results: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Combine and deduplicate results from multiple sources"""
        combined = {}
        
        for doc_id, doc, score in results:
            if doc_id not in combined:
                combined[doc_id] = (doc_id, doc, score)
            else:
                # Take the maximum score
                existing_score = combined[doc_id][2]
                combined[doc_id] = (doc_id, doc, max(existing_score, score))
        
        return list(combined.values())
    
    def expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        if not self.config.enable_query_expansion:
            return query
        
        # Simple query expansion with synonyms
        expansions = {
            'machine learning': ['ML', 'artificial intelligence', 'neural networks'],
            'AI': ['artificial intelligence', 'machine learning'],
            'attention': ['focus', 'mechanism', 'transformer'],
            'cache': ['memory', 'storage', 'buffer'],
            'retrieval': ['search', 'fetch', 'recovery'],
        }
        
        for term, synonyms in expansions.items():
            if term.lower() in query.lower():
                for synonym in synonyms:
                    if synonym.lower() not in query.lower():
                        query += f" {synonym}"
        
        return query