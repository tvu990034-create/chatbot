"""
knowledge_graph/entity_extraction.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Entity and Relation Extraction for Knowledge Graph construction.

Based on research papers:
- REBEL: Relation Extraction By End-to-end Language modeling
- Stanford OpenIE: Open Information Extraction
- spaCy for NER and relation extraction
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Entity and Relation Extractor for Knowledge Graph construction.
    
    Based on:
    - REBEL: https://github.com/Babelscape/rebel
    - Stanford OpenIE: https://github.com/stanfordnlp/openie
    
    Key features:
    - Named Entity Recognition (NER)
    - Relation extraction
    - Triple extraction (subject, relation, object)
    """
    
    def __init__(self):
        logger.info("EntityExtractor initialized")
    
    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[Dict[str, any]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            entity_types: Optional list of entity types to extract
        
        Returns:
            List of entity dictionaries
        """
        logger.info(f"Extracting entities from text (length: {len(text)})")
        
        # Simplified entity extraction
        # In practice, you'd use spaCy, transformers, or REBEL
        
        entities = []
        
        # Extract capitalized words as potential entities
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                entities.append({
                    "text": word,
                    "type": entity_types[0] if entity_types else "ENTITY",
                    "start": i,
                    "end": i + 1,
                })
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """
        Extract relations between entities.
        
        Args:
            text: Input text
            entities: List of extracted entities
        
        Returns:
            List of relation dictionaries
        """
        logger.info(f"Extracting relations from {len(entities)} entities")
        
        relations = []
        
        if len(entities) < 2:
            return relations
        
        # Extract relations between consecutive entities
        for i in range(len(entities) - 1):
            subject = entities[i]
            obj = entities[i + 1]
            
            # Find text between entities
            relation_text = text[
                subject["end"]:obj["start"]
            ].strip()
            
            if relation_text:
                relations.append({
                    "subject": subject["text"],
                    "relation": relation_text,
                    "object": obj["text"],
                    "confidence": 0.8,
                })
        
        logger.info(f"Extracted {len(relations)} relations")
        return relations
    
    def extract_triples(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        """
        Extract (subject, relation, object) triples from text.
        
        Args:
            text: Input text
        
        Returns:
            List of triple dictionaries
        """
        logger.info("Extracting triples from text")
        
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)
        
        triples = []
        for relation in relations:
            triples.append({
                "subject": relation["subject"],
                "relation": relation["relation"],
                "object": relation["object"],
            })
        
        logger.info(f"Extracted {len(triples)} triples")
        return triples