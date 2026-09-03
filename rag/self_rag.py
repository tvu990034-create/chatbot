"""
Self-RAG: Self-Reflective Retrieval-Augmented Generation
Based on AkariAsai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
https://arxiv.org/abs/2310.11511
"""

import json
from typing import Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReflectionType(str, Enum):
    """Types of reflections in Self-RAG."""
    IS_REL = "IsRel"      # Is the retrieved information relevant?
    IS_SUP = "IsSup"      # Is the retrieved information supportive?
    USEFUL = "Useful"    # Is the overall response useful?
    GRADE = "Grade"      # Overall quality assessment


@dataclass
class Reflection:
    """A single reflection on retrieval or generation."""
    reflection_type: ReflectionType
    content: str
    confidence: float = 0.0
    metadata: dict[str, Any] = None


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""
    content: str
    source: str
    score: float
    reflections: list[Reflection] = None


class SelfRAGProcessor:
    """Self-RAG processor with reflection capabilities."""
    
    def __init__(self, model: str):
        self.model = model
        self.reflection_history = []
        
    def retrieve_with_reflection(self, query: str, num_docs: int = 3) -> list[RetrievedDocument]:
        """Retrieve documents with self-reflection on relevance."""
        # In a full implementation, this would query a vector database
        # For now, we'll simulate with knowledge base
        
        docs = self._simulate_retrieval(query, num_docs)
        
        # Reflect on each retrieved document
        for doc in docs:
            doc.reflections = self._reflect_on_retrieval(query, doc)
        
        return docs
    
    def _simulate_retrieval(self, query: str, num_docs: int) -> list[RetrievedDocument]:
        """Simulate document retrieval (in production, use vector DB)."""
        # This would be replaced with actual vector search
        # For now, return placeholder documents
        
        knowledge_base = {
            "math": [
                "The derivative of x^n is n*x^(n-1) (power rule)",
                "The square root of a number x is a number y such that y*y = x",
                "The sum of angles in a triangle is 180 degrees"
            ],
            "science": [
                "The speed of light in vacuum is approximately 3x10^8 m/s",
                "Newton's first law states that objects at rest stay at rest",
                "The chemical symbol for gold is Au (atomic number 79)"
            ],
            "geography": [
                "The capital of France is Paris, known for the Eiffel Tower",
                "Jupiter is the largest planet in our solar system",
                "The Sahara Desert is located in Africa"
            ]
        }
        
        # Simple keyword matching to select category
        query_lower = query.lower()
        category = "science"  # default
        
        if any(word in query_lower for word in ['math', 'calculate', 'derivative', 'square']):
            category = "math"
        elif any(word in query_lower for word in ['capital', 'country', 'city', 'largest']):
            category = "geography"
        
        docs = []
        for i, content in enumerate(knowledge_base.get(category, [])[:num_docs]):
            docs.append(RetrievedDocument(
                content=content,
                source=f"{category}_kb_{i}",
                score=0.9 - (i * 0.1)  # Decreasing scores
                reflections=[]
            ))
        
        return docs
    
    def _reflect_on_retrieval(self, query: str, doc: RetrievedDocument) -> list[Reflection]:
        """Generate reflections on retrieved document."""
        reflections = []
        
        # IsRel reflection: Is this relevant to the query?
        is_rel = self._generate_reflection(
            query, 
            doc.content,
            "Is this information relevant to answering the question?"
        )
        reflections.append(is_rel)
        
        # IsSup reflection: Is this supportive of a good answer?
        is_sup = self._generate_reflection(
            query,
            doc.content,
            "Is this information helpful for constructing a correct answer?"
        )
        reflections.append(is_sup)
        
        return reflections
    
    def _generate_reflection(self, query: str, content: str, reflection_question: str) -> Reflection:
        """Generate a reflection using the model."""
        from litellm import completion
        
        prompt = f"""Query: {query}
Information: {content}
Question: {reflection_question}

Answer with YES or NO, then briefly explain."""
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0
            )
            
            reflection_text = response.choices[0].message.content.strip()
            
            # Determine confidence based on response
            confidence = 0.8 if "YES" in reflection_text.upper() else 0.3
            
            # Determine reflection type
            if "relevant" in reflection_question.lower():
                ref_type = ReflectionType.IS_REL
            elif "helpful" in reflection_question.lower() or "supportive" in reflection_question.lower():
                ref_type = ReflectionType.IS_SUP
            else:
                ref_type = ReflectionType.USEFUL
            
            return Reflection(
                reflection_type=ref_type,
                content=reflection_text,
                confidence=confidence,
                metadata={"query": query, "content": content}
            )
            
        except Exception as e:
            logger.warning(f"Reflection generation failed: {e}")
            return Reflection(
                reflection_type=ReflectionType.USEFUL,
                content="Error in reflection",
                confidence=0.0
            )
    
    def generate_with_rag(self, query: str, choices: List[str]) -> Tuple[str, List[Reflection]]:
        """Generate answer with RAG and self-reflection."""
        # Retrieve documents
        docs = self.retrieve_with_reflection(query)
        
        # Filter documents based on reflections
        relevant_docs = []
        for doc in docs:
            if doc.reflections:
                # Check if documents are marked as relevant
                is_relevant = any(
                    r.reflection_type == ReflectionType.IS_REL and r.confidence > 0.5
                    for r in doc.reflections
                )
                if is_relevant:
                    relevant_docs.append(doc)
        
        # Generate response using retrieved context
        context = "\n".join([doc.content for doc in relevant_docs]) if relevant_docs else ""
        
        answer = self._generate_rag_answer(query, choices, context)
        
        # Reflect on the overall response
        overall_reflection = self._reflect_on_generation(query, answer, context)
        
        return answer, [overall_reflection]
    
    def _generate_rag_answer(self, query: str, choices: List[str], context: str) -> str:
        """Generate answer using retrieved context."""
        from litellm import completion
        
        if context:
            prompt = f"""Context:
{context}

Question: {query}

Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Using the context above, answer the question. Respond with just the letter (A, B, C, or D)."""
        else:
            prompt = f"""Question: {query}
Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer with just the letter (A, B, C, or D)."""
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            
            answer = response.choices[0].message.content.strip().upper()
            for letter in ["A", "B", "C", "D"]:
                if letter in answer:
                    return letter
            return "A"
            
        except Exception as e:
            logger.warning(f"RAG answer generation failed: {e}")
            return "A"
    
    def _reflect_on_generation(self, query: str, answer: str, context: str) -> Reflection:
        """Reflect on the quality of the generated answer."""
        from litellm import completion
        
        prompt = f"""Question: {query}
My Answer: {answer}
Context Used: {context if context else "None"}

Is this answer likely to be correct? Answer YES or NO, then briefly explain."""
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0
            )
            
            reflection_text = response.choices[0].message.content.strip()
            confidence = 0.8 if "YES" in reflection_text.upper() else 0.3
            
            return Reflection(
                reflection_type=ReflectionType.GRADE,
                content=reflection_text,
                confidence=confidence,
                metadata={"query": query, "answer": answer}
            )
            
        except Exception as e:
            logger.warning(f"Generation reflection failed: {e}")
            return Reflection(
                reflection_type=ReflectionType.GRADE,
                content="Error in reflection",
                confidence=0.0
            )


class HybridRAGReasoning:
    """Combine Self-RAG with advanced reasoning for MMLU-Pro."""
    
    def __init__(self, model: str):
        self.model = model
        self.self_rag = SelfRAGProcessor(model)
        
    def answer_with_rag_reasoning(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """Answer using combined RAG and reasoning."""
        # Try RAG first
        rag_answer, reflections = self.self_rag.generate_with_rag(question, choices)
        
        # Check reflection confidence
        reflection_confidence = reflections[0].confidence if reflections else 0.0
        
        # If RAG confidence is low, use enhanced prompting
        if reflection_confidence < 0.5:
            reasoning_answer = self._enhanced_reasoning(question, choices)
            
            return {
                "answer": reasoning_answer,
                "method": "enhanced_reasoning",
                "confidence": 0.7,  # Fixed confidence for fallback
                "reflections": []
            }
        
        return {
            "answer": rag_answer,
            "method": "rag",
            "confidence": reflection_confidence,
            "reflections": [r.content for r in reflections]
        }
    
    def _enhanced_reasoning(self, question: str, choices: List[str]) -> str:
        """Enhanced reasoning fallback."""
        from litellm import completion
        
        prompt = f"""Think carefully to answer:

{question}

Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Consider each option step by step. Give your final answer as a letter."""
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.0
            )
            
            answer = response.choices[0].message.content.strip().upper()
            for letter in ["A", "B", "C", "D"]:
                if letter in answer:
                    return letter
            return "A"
            
        except Exception as e:
            logger.warning(f"Enhanced reasoning failed: {e}")
            return "A"