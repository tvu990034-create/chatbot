"""
knowledge_graph/kg_builder.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Knowledge Graph Builder for constructing KGs from text.

Based on research papers:
- COMET: Commonsense Transformers for Automatic Knowledge Graph Construction
  https://arxiv.org/abs/1906.05317
- LAMA: LAnguage Model Analysis of factual knowledge
  https://arxiv.org/abs/1909.01066
- REBEL: Relation Extraction By End-to-end Language modeling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Knowledge Graph Builder for constructing KGs from text.
    
    Based on:
    - COMET: https://arxiv.org/abs/1906.05317
    - REBEL: https://github.com/Babelscape/rebel
    
    Key features:
    - Entity extraction from text
    - Relation extraction between entities
    - Knowledge graph construction using NetworkX
    - Support for various storage formats
    """
    
    def __init__(
        self,
        kg_storage_path: Optional[Path] = None,
        kg_format: str = "networkx",
    ):
        self.kg_storage_path = kg_storage_path or settings.kg_storage_dir
        self.kg_format = kg_format
        
        # Initialize knowledge graph
        self.kg = nx.DiGraph()
        
        logger.info(
            f"KnowledgeGraphBuilder initialized with storage_path={kg_storage_path}, "
            f"format={kg_format}"
        )
    
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
            List of entity dictionaries with text, type, and position
        """
        # Simplified entity extraction
        # In practice, you'd use spaCy, REBEL, or other NER tools
        
        entities = []
        
        # Placeholder implementation
        # Split by common delimiters to find potential entities
        words = text.split()
        
        for i, word in enumerate(words):
            # Capitalized words might be entities
            if word[0].isupper() and len(word) > 2:
                entities.append({
                    "text": word,
                    "type": entity_types[0] if entity_types else "ENTITY",
                    "start": i,
                    "end": i + 1,
                })
        
        logger.info(f"Extracted {len(entities)} entities from text")
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
            List of relation dictionaries (subject, relation, object)
        """
        relations = []
        
        # Simplified relation extraction
        # In practice, you'd use REBEL, OpenIE, or relation extraction models
        
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
                    "confidence": 0.8,  # Placeholder confidence
                })
        
        logger.info(f"Extracted {len(relations)} relations from text")
        return relations
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: Optional[Dict[str, any]] = None,
    ):
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            attributes: Optional entity attributes
        """
        attributes = attributes or {}
        attributes["type"] = entity_type
        
        self.kg.add_node(entity_id, **attributes)
    
    def add_relation(
        self,
        subject: str,
        relation: str,
        obj: str,
        attributes: Optional[Dict[str, any]] = None,
    ):
        """
        Add a relation to the knowledge graph.
        
        Args:
            subject: Subject entity ID
            relation: Relation type
            obj: Object entity ID
            attributes: Optional relation attributes
        """
        attributes = attributes or {}
        
        self.kg.add_edge(subject, obj, relation=relation, **attributes)
    
    def build_from_text(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> nx.DiGraph:
        """
        Build knowledge graph from text.
        
        Args:
            text: Input text
            entity_types: Optional list of entity types
        
        Returns:
            NetworkX DiGraph representing the knowledge graph
        """
        logger.info("Building knowledge graph from text")
        
        # Extract entities
        entities = self.extract_entities(text, entity_types)
        
        # Extract relations
        relations = self.extract_relations(text, entities)
        
        # Build graph
        for entity in entities:
            entity_id = entity["text"].lower().replace(" ", "_")
            self.add_entity(entity_id, entity["type"])
        
        for relation in relations:
            subject_id = relation["subject"].lower().replace(" ", "_")
            object_id = relation["object"].lower().replace(" ", "_")
            self.add_relation(
                subject_id,
                relation["relation"],
                object_id,
                {"confidence": relation["confidence"]}
            )
        
        logger.info(f"Built knowledge graph with {self.kg.number_of_nodes()} nodes "
                    f"and {self.kg.number_of_edges()} edges")
        
        return self.kg
    
    def build_from_triples(
        self,
        triples: List[Tuple[str, str, str]],
    ) -> nx.DiGraph:
        """
        Build knowledge graph from triples.
        
        Args:
            triples: List of (subject, relation, object) tuples
        
        Returns:
            NetworkX DiGraph representing the knowledge graph
        """
        logger.info(f"Building knowledge graph from {len(triples)} triples")
        
        for subject, relation, obj in triples:
            # Add entities if not present
            if subject not in self.kg:
                self.add_entity(subject, "ENTITY")
            if obj not in self.kg:
                self.add_entity(obj, "ENTITY")
            
            # Add relation
            self.add_relation(subject, relation, obj)
        
        logger.info(f"Built knowledge graph with {self.kg.number_of_nodes()} nodes "
                    f"and {self.kg.number_of_edges()} edges")
        
        return self.kg
    
    def save_kg(
        self,
        output_path: Optional[Path] = None,
        format: Optional[str] = None,
    ):
        """
        Save knowledge graph to file.
        
        Args:
            output_path: Output file path
            format: Output format (networkx, gexf, graphml, json)
        """
        output_path = output_path or self.kg_storage_path
        format = format or self.kg_format
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving knowledge graph to {output_path}")
        
        if format == "networkx":
            import pickle
            with open(output_path, "wb") as f:
                pickle.dump(self.kg, f)
        elif format == "gexf":
            nx.write_gexf(self.kg, output_path)
        elif format == "graphml":
            nx.write_graphml(self.kg, output_path)
        elif format == "json":
            from networkx.readwrite import json_graph
            data = json_graph.node_link_data(self.kg)
            import json
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info("Knowledge graph saved successfully")
    
    def load_kg(
        self,
        input_path: Path,
        format: Optional[str] = None,
    ) -> nx.DiGraph:
        """
        Load knowledge graph from file.
        
        Args:
            input_path: Input file path
            format: Input format
        
        Returns:
            Loaded knowledge graph
        """
        format = format or self.kg_format
        
        logger.info(f"Loading knowledge graph from {input_path}")
        
        if format == "networkx":
            import pickle
            with open(input_path, "rb") as f:
                self.kg = pickle.load(f)
        elif format == "gexf":
            self.kg = nx.read_gexf(input_path)
        elif format == "graphml":
            self.kg = nx.read_graphml(input_path)
        elif format == "json":
            from networkx.readwrite import json_graph
            import json
            with open(input_path, "r") as f:
                data = json.load(f)
            self.kg = json_graph.node_link_graph(data)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Loaded knowledge graph with {self.kg.number_of_nodes()} nodes "
                    f"and {self.kg.number_of_edges()} edges")
        
        return self.kg
    
    def get_kg_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with KG statistics
        """
        return {
            "num_nodes": self.kg.number_of_nodes(),
            "num_edges": self.kg.number_of_edges(),
            "avg_degree": sum(dict(self.kg.degree()).values()) / self.kg.number_of_nodes() if self.kg.number_of_nodes() > 0 else 0,
            "density": nx.density(self.kg),
            "is_connected": nx.is_weakly_connected(self.kg),
        }