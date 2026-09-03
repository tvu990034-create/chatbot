"""
knowledge_graph/graph_rag.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
GraphRAG implementation for graph-based retrieval-augmented generation.

Based on research papers:
- GraphRAG: Microsoft's Graph-based RAG
- LightRAG: Simple and Fast Retrieval-Augmented Generation
- Graph-of-Thought: https://arxiv.org/abs/2308.09687
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from knowledge_graph.kg_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class GraphRAG:
    """
    GraphRAG implementation for graph-based retrieval.
    
    Based on:
    - Microsoft GraphRAG: https://github.com/microsoft/graphrag
    - LightRAG: https://github.com/HKUDS/LightRAG
    
    Key features:
    - Entity-based retrieval from knowledge graph
    - Multi-hop reasoning over graph structure
    - Community detection for better retrieval
    - Integration with RAG pipeline
    """
    
    def __init__(
        self,
        kg: Optional[nx.DiGraph] = None,
        kg_builder: Optional[KnowledgeGraphBuilder] = None,
    ):
        self.kg = kg
        self.kg_builder = kg_builder or KnowledgeGraphBuilder()
        
        if kg is None and kg_builder is not None:
            self.kg = kg_builder.kg
        
        logger.info("GraphRAG initialized")
    
    def retrieve_by_entity(
        self,
        entity: str,
        top_k: int = 5,
    ) -> List[Dict[str, any]]:
        """
        Retrieve information related to an entity.
        
        Args:
            entity: Entity name or ID
            top_k: Number of results to return
        
        Returns:
            List of related entities and relations
        """
        if self.kg is None:
            logger.warning("Knowledge graph not loaded")
            return []
        
        logger.info(f"Retrieving information for entity: {entity}")
        
        # Find entity in graph
        entity_id = entity.lower().replace(" ", "_")
        
        if entity_id not in self.kg:
            logger.warning(f"Entity {entity} not found in knowledge graph")
            return []
        
        # Get neighbors
        neighbors = list(self.kg.neighbors(entity_id))
        
        # Get relations
        relations = []
        for neighbor in neighbors[:top_k]:
            edge_data = self.kg.get_edge_data(entity_id, neighbor)
            relations.append({
                "subject": entity_id,
                "relation": edge_data.get("relation", "related_to"),
                "object": neighbor,
                "confidence": edge_data.get("confidence", 1.0),
            })
        
        return relations
    
    def multi_hop_retrieval(
        self,
        entity: str,
        max_hops: int = 2,
        top_k: int = 10,
    ) -> List[Dict[str, any]]:
        """
        Perform multi-hop retrieval over the knowledge graph.
        
        Args:
            entity: Starting entity
            max_hops: Maximum number of hops
            top_k: Number of results to return
        
        Returns:
            List of multi-hop paths
        """
        if self.kg is None:
            logger.warning("Knowledge graph not loaded")
            return []
        
        logger.info(f"Multi-hop retrieval from {entity} with max_hops={max_hops}")
        
        entity_id = entity.lower().replace(" ", "_")
        
        if entity_id not in self.kg:
            return []
        
        # Perform BFS up to max_hops
        paths = []
        visited = {entity_id}
        queue = [(entity_id, [entity_id])]
        
        while queue and len(paths) < top_k:
            current, path = queue.pop(0)
            
            if len(path) > max_hops + 1:
                continue
            
            for neighbor in self.kg.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    
                    if len(new_path) > 1:
                        edge_data = self.kg.get_edge_data(current, neighbor)
                        paths.append({
                            "path": new_path,
                            "relation": edge_data.get("relation", "related_to"),
                            "length": len(new_path) - 1,
                        })
                    
                    queue.append((neighbor, new_path))
        
        return paths[:top_k]
    
    def community_detection(
        self,
        resolution: float = 1.0,
    ) -> Dict[str, List[str]]:
        """
        Detect communities in the knowledge graph.
        
        Args:
            resolution: Resolution parameter for community detection
        
        Returns:
            Dictionary mapping community IDs to entity lists
        """
        if self.kg is None:
            logger.warning("Knowledge graph not loaded")
            return {}
        
        logger.info("Detecting communities in knowledge graph")
        
        try:
            # Use Louvain community detection
            from networkx.algorithms import community
            communities = community.louvain_communities(
                self.kg.to_undirected(),
                resolution=resolution
            )
            
            # Convert to dictionary
            community_dict = {}
            for i, comm in enumerate(communities):
                community_dict[f"community_{i}"] = list(comm)
            
            logger.info(f"Detected {len(community_dict)} communities")
            return community_dict
        
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            return {}
    
    def retrieve_by_community(
        self,
        query_entity: str,
        top_k: int = 5,
    ) -> List[Dict[str, any]]:
        """
        Retrieve information from the same community as query entity.
        
        Args:
            query_entity: Query entity
            top_k: Number of results to return
        
        Returns:
            List of entities from the same community
        """
        communities = self.community_detection()
        
        if not communities:
            return []
        
        # Find community of query entity
        entity_id = query_entity.lower().replace(" ", "_")
        target_community = None
        
        for comm_id, entities in communities.items():
            if entity_id in entities:
                target_community = entities
                break
        
        if target_community is None:
            logger.warning(f"Entity {query_entity} not found in any community")
            return []
        
        # Return other entities from the same community
        results = []
        for entity in target_community:
            if entity != entity_id:
                results.append({
                    "entity": entity,
                    "community": "same",
                })
        
        return results[:top_k]