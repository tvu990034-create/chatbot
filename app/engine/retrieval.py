"""
Retrieval Engine - Mathematical equations for document retrieval
Extracted from PDF files - these are the ONLY equations that matter
"""

import math
import time
from typing import List, Tuple, Dict, Any

class RetrievalEngine:
    """
    Implements ALL retrieval equations from PDF files
    These equations are the single source of truth for retrieval logic
    """
    
    def __init__(self):
        # Equation parameters from PDFs
        self.encoder_model = 'all-MiniLM-L6-v2'
        self.dim = 384
        self.nlist = 4096
        self.pq_bytes = 8
        self.subkey_size = 512
        self.similarity_threshold = 0.92
        self.t_coarse_ratio = 0.2
        self.t_pq_ratio = 0.8
    
    # EQUATION: t_corpus = (time.time() - start) * 1000 # ms
    def calc_t_corpus(self, start: float) -> float:
        """t_corpus = (time.time() - start) * 1000 # ms"""
        return (time.time() - start) * 1000
    
    # EQUATION: t_search_raw = (time.time() - t1) * 1000
    def calc_t_search_raw(self, t1: float) -> float:
        """t_search_raw = (time.time() - t1) * 1000"""
        return (time.time() - t1) * 1000
    
    # EQUATION: t_coarse = t_search_total * 0.2
    def calc_t_coarse(self, t_search_total: float) -> float:
        """t_coarse = t_search_total * 0.2"""
        return t_search_total * self.t_coarse_ratio
    
    # EQUATION: t_pq = t_search_total * 0.8
    def calc_t_pq(self, t_search_total: float) -> float:
        """t_pq = t_search_total * 0.8"""
        return t_search_total * self.t_pq_ratio
    
    # EQUATION: t_meta = (time.time() - t2) * 1000
    def calc_t_meta(self, t2: float) -> float:
        """t_meta = (time.time() - t2) * 1000"""
        return (time.time() - t2) * 1000
    
    # EQUATION: successful = [r for r in results if r.top_score >= 0.0]
    def calc_successful(self, results: List[Dict]) -> List[Dict]:
        """successful = [r for r in results if r.top_score >= 0.0]"""
        return [r for r in results if r.get('top_score', -1) >= 0.0]
    
    # EQUATION: top = chunks[0].score if chunks else 0.0
    def calc_top(self, chunks: List[Dict]) -> float:
        """top = chunks[0].score if chunks else 0.0"""
        return chunks[0].get('score', 0.0) if chunks else 0.0
    
    # EQUATION: score = min(base + boost, 1.0)
    def calc_score(self, base: float, boost: float) -> float:
        """score = min(base + boost, 1.0)"""
        return min(base + boost, 1.0)
    
    # EQUATION: boost = min(len(ent.text) / 20, 0.2)
    def calc_boost(self, text_length: int) -> float:
        """boost = min(len(ent.text) / 20, 0.2)"""
        return min(text_length / 20, 0.2)
    
    # EQUATION: base = 0.8 if ent.label_ in ("PERSON", "GPE", "ORG") else 0.6
    def calc_base(self, label: str) -> float:
        """base = 0.8 if ent.label_ in ("PERSON", "GPE", "ORG") else 0.6"""
        return 0.8 if label in ("PERSON", "GPE", "ORG") else 0.6
    
    # EQUATION: q_emb = self.encoder.encode([query], show_progress_bar=False)[0]
    def encode_query(self, query: str, encoder) -> List[float]:
        """q_emb = self.encoder.encode([query], show_progress_bar=False)[0]"""
        return encoder.encode([query], show_progress_bar=False)[0]
    
    # EQUATION: distances, ids = self.index.search(embedding.reshape(1, -1), 1)
    def search_index(self, index, embedding: List[float], k: int = 1) -> Tuple[List, List]:
        """distances, ids = self.index.search(embedding.reshape(1, -1), k)"""
        import numpy as np
        return index.search(np.array(embedding).reshape(1, -1), k)
    
    # EQUATION: results = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if s >= threshold]
    def filter_results(self, ids: List, scores: List, threshold: float = 0.0) -> List[Tuple[int, float]]:
        """results = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if s >= threshold]"""
        return [(int(i), float(s)) for i, s in zip(ids, scores) if s >= threshold]
    
    # EQUATION: t_q = (time.time() - t0) * 1000
    def calc_t_q(self, t0: float) -> float:
        """t_q = (time.time() - t0) * 1000"""
        return (time.time() - t0) * 1000
    
    # EQUATION: t_search_total = (time.time() - t1) * 1000
    def calc_t_search_total(self, t1: float) -> float:
        """t_search_total = (time.time() - t1) * 1000"""
        return (time.time() - t1) * 1000
    
    # MASTER EQUATION: Complete retrieval timing
    def calculate_retrieval_timing(self, query: str, encoder, index, k: int = 5) -> Dict[str, Any]:
        """
        MASTER EQUATION: Combines all retrieval equations
        Returns complete timing and results
        """
        t0 = time.time()
        
        # Encode query
        q_emb = self.encode_query(query, encoder)
        t_q = self.calc_t_q(t0)
        
        t1 = time.time()
        
        # Search index
        distances, ids = self.search_index(index, q_emb, k)
        
        # Calculate coarse and PQ times
        t_search_total = self.calc_t_search_total(t1)
        t_coarse = self.calc_t_coarse(t_search_total)
        t_pq = self.calc_t_pq(t_search_total)
        
        # Filter results
        t2 = time.time()
        results = self.filter_results(ids[0], distances[0], self.similarity_threshold)
        t_meta = self.calc_t_meta(t2)
        
        # Calculate top score
        successful = self.calc_successful([{'top_score': s} for _, s in results])
        top = self.calc_top(successful) if successful else 0.0
        
        return {
            "t_query_enc_ms": t_q,
            "t_coarse_ms": t_coarse,
            "t_pq_ms": t_pq,
            "t_meta_ms": t_meta,
            "t_search_total_ms": t_search_total,
            "results": results,
            "top_score": top,
            "num_results": len(results)
        }


# Global retrieval engine instance
retrieval_engine = RetrievalEngine()
