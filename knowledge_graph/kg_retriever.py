"""
knowledge_graph/kg_retriever.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Knowledge Graph Retriever for querying KGs.

Based on research papers:
- QA-GNN: https://arxiv.org/abs/2104.06378
- JointLK: https://arxiv.org/abs/2110.09829
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraphRetriever:
    """
    Knowledge Graph Retriever for querying KGs.
    
    Based on:
    - QA-GNN: https://arxiv.org/abs/2104.06378
    - JointLK: https://arxiv.org/abs/2110.09829
    
    Key features:
    - SPARQL-like querying over knowledge graph
    - Path-based retrieval
    - Subgraph extraction
    """
    
    def __init__(self, kg: nx.DiGraph):
        self.kg = kg
        logger.info("KnowledgeGraphRetriever initialized")
    
    def query_entity(
        self,
        entity: str,
    ) -> Dict[str, any]:
        """
        Query information about a specific entity.
        
        Args:
            entity: Entity name or ID
        
        Returns:
            Entity information dictionary
        """
        entity_id = entity.lower().replace(" ", "_")
        
        if entity_id not in self.kg:
            return {"error": f"Entity {entity} not found"}
        
        node_data = self.kg.nodes[entity_id]
        neighbors = list(self.kg.neighbors(entity_id))
        
        return {
            "entity": entity_id,
            "attributes": node_data,
            "neighbors": neighbors,
            "num_relations": len(neighbors),
        }
    
    def query_relation(
        self,
        subject: str,
        relation: str,
    ) -> List[Dict[str, any]]:
        """
        Query objects connected by a specific relation.
        
        Args:
            subject: Subject entity
            relation: Relation type
        
        Returns:
            List of connected objects
        """
        subject_id = subject.lower().replace(" ", "_")
        
        if subject_id not in self.kg:
            return []
        
        results = []
        for neighbor in self.kg.neighbors(subject_id):
            edge_data = self.kg.get_edge_data(subject_id, neighbor)
            if edge_data.get("relation") == relation:
                results.append({
                    "object": neighbor,
                    "attributes": edge_data,
                })
        
        return results
    
    def find_path(
        self,
        source: str,
        target: str,
        max_length: int = 5,
    ) -> List[List[str]]:
        """
        Find paths between two entities.
        
        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length
        
        Returns:
            List of paths
        """
        source_id = source.lower().replace(" ", "_")
        target_id = target.lower().replace(" ", "_")
        
        if source_id not in self.kg or target_id not in self.kg:
            return []
        
        try:
            paths = list(nx.all_simple_paths(
                self.kg,
                source_id,
                target_id,
                cutoff=max_length
            ))
            return paths
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return []
    
    def extract_subgraph(
        self,
        entities: List[str],
        radius: int = 1,
    ) -> nx.DiGraph:
        """
        Extract a subgraph around entities.
        
        Args:
            entities: List of central entities
            radius: Radius for subgraph extraction
        
        Returns:
            Subgraph as NetworkX DiGraph
        """
        entity_ids = [e.lower().replace(" ", "_") for e in entities]
        
        # Collect all nodes within radius
        nodes = set(entity_ids)
        for _ in range(radius):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.kg.neighbors(node))
            nodes.update(new_nodes)
        
        # Extract subgraph
        subgraph = self.kg.subgraph(nodes).copy()
        
        return subgraph