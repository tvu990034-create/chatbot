"""
Knowledge Base Quality Filter
Removes low-quality and duplicate documents to improve retrieval quality
"""
import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class KnowledgeBaseFilter:
    """Filters and improves knowledge base quality"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_length': 20,
            'max_length': 2000,
            'min_meaningful_words': 3,
            'max_repetition_ratio': 0.5
        }
    
    def filter_documents(self, documents: List[str], doc_ids: List[str]) -> Tuple[List[str], List[str]]:
        """Filter documents based on quality metrics"""
        logger.info(f"Filtering {len(documents)} documents...")
        
        filtered_docs = []
        filtered_ids = []
        stats = {
            'total': len(documents),
            'too_short': 0,
            'too_long': 0,
            'low_quality': 0,
            'duplicate': 0,
            'kept': 0
        }
        
        seen_content = set()
        
        for doc, doc_id in zip(documents, doc_ids):
            # Check for duplicates
            content_hash = self._compute_content_hash(doc)
            if content_hash in seen_content:
                stats['duplicate'] += 1
                continue
            seen_content.add(content_hash)
            
            # Check quality
            quality_score = self._assess_quality(doc)
            
            if quality_score < 0.5:  # Quality threshold
                if len(doc) < self.quality_thresholds['min_length']:
                    stats['too_short'] += 1
                elif len(doc) > self.quality_thresholds['max_length']:
                    stats['too_long'] += 1
                else:
                    stats['low_quality'] += 1
                continue
            
            # Keep high-quality documents
            filtered_docs.append(doc)
            filtered_ids.append(doc_id)
            stats['kept'] += 1
        
        logger.info(f"Filtering complete: {stats}")
        return filtered_docs, filtered_ids
    
    def _compute_content_hash(self, text: str) -> str:
        """Compute hash for duplicate detection"""
        import hashlib
        # Normalize for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _assess_quality(self, text: str) -> float:
        """Assess document quality (0-1 score)"""
        score = 1.0
        
        # Length penalty
        length = len(text)
        if length < self.quality_thresholds['min_length']:
            score *= 0.1
        elif length > self.quality_thresholds['max_length']:
            score *= 0.5
        
        # Repetition penalty
        repetition_ratio = self._compute_repetition_ratio(text)
        if repetition_ratio > self.quality_thresholds['max_repetition_ratio']:
            score *= 0.3
        
        # Meaningful words check
        words = text.split()
        meaningful_words = [w for w in words if len(w) > 3]
        if len(meaningful_words) < self.quality_thresholds['min_meaningful_words']:
            score *= 0.2
        
        # Check for technical content (bonus)
        technical_keywords = ['algorithm', 'model', 'system', 'data', 'neural', 'learning', 
                             'network', 'attention', 'cache', 'processing', 'analysis']
        if any(keyword in text.lower() for keyword in technical_keywords):
            score *= 1.5
        
        # Check for garbage content (penalty)
        garbage_patterns = [r'\d{10,}', r'[a-z]{10,}']  # Long numbers/letter sequences
        for pattern in garbage_patterns:
            if re.search(pattern, text):
                score *= 0.5
                break
        
        return min(score, 1.0)
    
    def _compute_repetition_ratio(self, text: str) -> float:
        """Compute ratio of repeated content"""
        words = text.lower().split()
        if len(words) < 5:
            return 0.0
        
        unique_words = set(words)
        repetition = 1.0 - (len(unique_words) / len(words))
        return repetition
    
    def prioritize_tech_content(self, documents: List[str], doc_ids: List[str]) -> Tuple[List[str], List[str]]:
        """Prioritize technical content for better retrieval"""
        logger.info("Prioritizing technical content...")
        
        # Expanded tech keywords to better identify ML/AI content
        tech_keywords = [
            'machine learning', 'artificial intelligence', 'neural network', 'deep learning',
            'attention mechanism', 'transformer', 'kv cache', 'retrieval', 'embedding',
            'algorithm', 'optimization', 'quantization', 'speculative decoding',
            'gradient', 'backpropagation', 'loss function', 'training', 'inference',
            'tensor', 'vector', 'matrix', 'parameter', 'model', 'dataset'
        ]
        
        prioritized_docs = []
        prioritized_ids = []
        other_docs = []
        other_ids = []
        
        for doc, doc_id in zip(documents, doc_ids):
            doc_lower = doc.lower()
            is_tech = any(keyword in doc_lower for keyword in tech_keywords)
            
            if is_tech:
                prioritized_docs.append(doc)
                prioritized_ids.append(doc_id)
            else:
                other_docs.append(doc)
                other_ids.append(doc_id)
        
        logger.info(f"Technical documents: {len(prioritized_docs)}, Other: {len(other_docs)}")
        
        # Combine with tech content first
        combined_docs = prioritized_docs + other_docs
        combined_ids = prioritized_ids + other_ids
        
        return combined_docs, combined_ids