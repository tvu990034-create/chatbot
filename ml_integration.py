"""
Simple Machine Learning Integration for Query Prediction
Basic ML-based query pattern recognition and prediction
"""

import json
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

class QueryPatternML:
    """Simple ML-based query pattern recognition."""
    
    def __init__(self, data_file: str = "ml_data/query_patterns.json"):
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(exist_ok=True)
        
        self.query_history = []
        self.pattern_weights = defaultdict(float)
        self.query_categories = Counter()
        
        self._load_training_data()
    
    def _load_training_data(self):
        """Load training data from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.query_history = data.get("query_history", [])
                    self.pattern_weights = defaultdict(float, data.get("pattern_weights", {}))
                    self.query_categories = Counter(data.get("query_categories", {}))
            except:
                pass
    
    def _save_training_data(self):
        """Save training data to file."""
        data = {
            "query_history": self.query_history[-100:],  # Keep last 100 queries
            "pattern_weights": dict(self.pattern_weights),
            "query_categories": dict(self.query_categories),
            "last_updated": datetime.now().isoformat()
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_query(self, query: str, response_time: float, cache_hit: bool):
        """Record a query for ML training."""
        query_lower = query.lower()
        
        # Extract features
        features = self._extract_features(query_lower)
        
        # Store query with metadata
        self.query_history.append({
            "query": query,
            "features": features,
            "response_time": response_time,
            "cache_hit": cache_hit,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update pattern weights
        for feature in features:
            self.pattern_weights[feature] += 1.0
        
        # Update category counts
        category = self._classify_query_simple(query_lower)
        self.query_categories[category] += 1
        
        # Save periodically
        if len(self.query_history) % 10 == 0:
            self._save_training_data()
    
    def _extract_features(self, query: str) -> List[str]:
        """Extract features from query for ML processing."""
        features = []
        
        # Word n-grams (1-grams and 2-grams)
        words = query.split()
        
        # Single words (1-grams)
        for word in words:
            if len(word) > 2:  # Skip very short words
                features.append(f"word_{word}")
        
        # Word pairs (2-grams)
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                features.append(f"pair_{words[i]}_{words[i+1]}")
        
        # Query length features
        if len(words) <= 3:
            features.append("length_short")
        elif len(words) <= 6:
            features.append("length_medium")
        else:
            features.append("length_long")
        
        # Common patterns
        if any(word in query for word in ["what", "how", "why", "explain"]):
            features.append("question_type")
        if any(word in query for word in ["code", "function", "python", "write"]):
            features.append("coding_task")
        if any(word in query for word in ["calculate", "math", "solve", "equation"]):
            features.append("math_task")
        
        return features
    
    def _classify_query_simple(self, query: str) -> str:
        """Simple query classification."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        elif any(word in query_lower for word in ["code", "function", "python", "algorithm"]):
            return "coding"
        elif any(word in query_lower for word in ["calculate", "math", "equation", "solve"]):
            return "math"
        elif any(word in query_lower for word in ["explain", "what", "how", "why"]):
            return "explanation"
        else:
            return "general"
    
    def predict_next_queries(self, current_query: str, num_predictions: int = 5) -> List[Dict[str, Any]]:
        """Predict likely next queries based on patterns."""
        current_features = self._extract_features(current_query.lower())
        
        # Score previous queries based on feature overlap
        scored_queries = []
        
        for prev_query in self.query_history[-50:]:  # Consider last 50 queries
            prev_features = set(prev_query["features"])
            current_features_set = set(current_features)
            
            # Calculate feature overlap
            overlap = len(prev_features & current_features_set)
            
            if overlap > 0:
                scored_queries.append({
                    "query": prev_query["query"],
                    "score": overlap,
                    "response_time": prev_query["response_time"],
                    "cache_hit": prev_query["cache_hit"]
                })
        
        # Sort by score and return top predictions
        scored_queries.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_queries[:num_predictions]
    
    def get_hot_patterns(self) -> List[Dict[str, Any]]:
        """Get currently hot query patterns."""
        # Sort patterns by weight
        sorted_patterns = sorted(
            self.pattern_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"pattern": pattern, "weight": weight}
            for pattern, weight in sorted_patterns[:20]
        ]
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring."""
        if not self.query_history:
            return {}
        
        recent_queries = self.query_history[-100:]
        
        response_times = [q["response_time"] for q in recent_queries]
        cache_hits = sum(1 for q in recent_queries if q["cache_hit"])
        
        return {
            "total_queries": len(self.query_history),
            "recent_queries": len(recent_queries),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "cache_hit_rate": cache_hits / len(recent_queries) if recent_queries else 0,
            "query_categories": dict(self.query_categories),
            "hot_patterns": self.get_hot_patterns()[:10]
        }

# Global ML instance
_query_ml = None

def get_query_ml() -> QueryPatternML:
    """Get global query ML instance."""
    global _query_ml
    if _query_ml is None:
        _query_ml = QueryPatternML()
    return _query_ml

# Integration with main API
if __name__ == "__main__":
    ml = QueryPatternML()
    
    # Simulate some queries
    ml.record_query("hello", 2.5, False)
    ml.record_query("write python code", 4.2, False)
    ml.record_query("what is ai", 3.1, False)
    ml.record_query("calculate 2+2", 2.8, True)
    
    # Get predictions
    predictions = ml.predict_next_queries("write code")
    print(f"Predictions: {predictions}")
    
    # Get statistics
    stats = ml.get_usage_statistics()
    print(f"Statistics: {stats}")