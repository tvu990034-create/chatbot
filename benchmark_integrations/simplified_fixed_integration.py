"""
Simplified Fixed Integration Layer
Minimal gateway wrapper for retry logic and timeout handling
"""

import time
from typing import Dict, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gateway.litellm_gateway import chat as gateway_chat

# No domain enhancements - pure baseline

class SimplifiedFixedIntegrationLayer:
    """
    Minimal gateway integration - retry logic and timeout handling only.
    """
    
    def __init__(self):
        self.enhancement_stats = {
            "total_queries": 0,
            "total_time": 0
        }
    
    def query_with_gateway(self, query: str, model: str = "ollama/phi3:mini") -> Tuple[str, Dict]:
        """Query using minimal gateway - retry logic and timeout handling."""
        start_time = time.time()
        
        enhancement_info = {
            "original_query": query,
            "enhancements_applied": ["minimal_gateway"],
            "model": model,
            "domain_enhancements": []
        }
        
        # Gateway call with retry logic
        reply = gateway_chat(
            messages=[{"role": "user", "content": query}],
            model=model
        )
        
        elapsed = time.time() - start_time
        
        enhancement_info["response_time"] = elapsed
        enhancement_info["enhanced_query"] = query
        self.enhancement_stats["total_queries"] += 1
        self.enhancement_stats["total_time"] += elapsed
        
        return reply, enhancement_info
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.enhancement_stats