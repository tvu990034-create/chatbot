"""
Simple Knowledge Base for RAG (Retrieval-Augmented Generation)
Stores recent facts and information to address knowledge cutoff limitations
"""

import re
from typing import List, Dict, Tuple

class SimpleKnowledgeBase:
    """Simple in-memory knowledge base for recent information."""
    
    def __init__(self):
        self.facts = {
            "technology": [
                "As of 2024, GPT-4 and Claude-3 are among the most advanced LLMs",
                "Ollama is a local LLM runner that supports models like phi3:mini",
                "phi3:mini is a 4B parameter model by Microsoft released in 2024",
                "phi3:mini is optimized for efficiency and performance on edge devices",
                "Local LLMs provide privacy and cost benefits compared to cloud APIs",
            ],
            "science": [
                "James Webb Space Telescope launched in December 2021",
                "COVID-19 pandemic declared in March 2020",
                "Perseverance rover landed on Mars in February 2021",
                "AI breakthroughs in 2024 include improved reasoning and multimodal capabilities",
                "Quantum computing advances continue in 2024 with new qubit records",
            ],
            "general": [
                "Current year is 2024",
                "Python 3.12 was released in October 2023",
                "AI safety and alignment research remains a priority in 2024",
                "Climate change continues to be a major global challenge in 2024",
            ],
            "computing": [
                "phi3:mini uses transformer architecture with optimizations",
                "Edge AI deployment is growing in 2024 for privacy and latency reasons",
                "Model compression techniques like quantization enable local deployment",
                "GPU acceleration significantly improves LLM inference speed",
            ]
        }
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search knowledge base for relevant facts using simple keyword matching.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of relevant facts
        """
        query_lower = query.lower()
        relevant_facts = []
        
        for category, facts in self.facts.items():
            for fact in facts:
                # Simple keyword matching
                fact_lower = fact.lower()
                # Check if any word in query appears in fact
                query_words = set(re.findall(r'\w+', query_lower))
                fact_words = set(re.findall(r'\w+', fact_lower))
                
                # Calculate overlap
                overlap = len(query_words & fact_words)
                
                if overlap > 0:
                    relevant_facts.append((overlap, fact))
        
        # Sort by overlap and return top_k
        relevant_facts.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in relevant_facts[:top_k]]
    
    def add_fact(self, category: str, fact: str):
        """Add a new fact to the knowledge base."""
        if category not in self.facts:
            self.facts[category] = []
        self.facts[category].append(fact)


# Global instance
_knowledge_base = SimpleKnowledgeBase()


def get_knowledge_base() -> SimpleKnowledgeBase:
    """Get the global knowledge base instance."""
    return _knowledge_base


def retrieve_context(query: str) -> str:
    """
    Retrieve relevant context for a query.
    
    Args:
        query: The user's query
    
    Returns:
        Context string with relevant facts
    """
    kb = get_knowledge_base()
    relevant_facts = kb.search(query, top_k=2)
    
    if relevant_facts:
        context = "Relevant information:\n" + "\n".join(f"- {fact}" for fact in relevant_facts)
        return context
    return ""
