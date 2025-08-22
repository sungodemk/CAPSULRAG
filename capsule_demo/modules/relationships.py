"""
Relationship mining module for CapsuleRAG: cross-capsule relationships and entity networks.
"""

import threading
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from .utils import extract_entities_demo


class RelationshipMiner:
    """Mine relationships between capsules and build entity networks."""
    
    def __init__(self):
        self.relationships: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.entity_co_occurrences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.lock = threading.RLock()
    
    def analyze_relationships(self, doc_id: int, entities: Dict[str, List[str]]):
        """Analyze relationships between entities in a document."""
        with self.lock:
            # Record entity co-occurrences within the same document
            all_entities = []
            for entity_type, entity_list in entities.items():
                all_entities.extend([(entity, entity_type) for entity in entity_list])
            
            # Create co-occurrence matrix
            for i, (entity1, type1) in enumerate(all_entities):
                for j, (entity2, type2) in enumerate(all_entities):
                    if i != j:
                        key = f"{type1}:{entity1}"
                        related_key = f"{type2}:{entity2}"
                        self.entity_co_occurrences[key][related_key] += 1
    
    def find_related_capsules(self, doc_id: int, docs: Dict, doc_filenames: Dict, 
                            similarity_threshold: float = 0.3) -> List[Tuple[int, float, str]]:
        """Find capsules related to the given document."""
        if doc_id not in docs:
            return []
        
        related = []
        
        try:
            # Get entities for this document
            current_entities = extract_entities_demo(docs[doc_id], doc_filenames.get(doc_id, ""))
            current_entity_set = set()
            for entities in current_entities.values():
                current_entity_set.update(entities)
            
            if not current_entity_set:
                return []
            
            # Compare with other documents
            for other_doc_id in docs.keys():
                if other_doc_id == doc_id:
                    continue
                
                other_entities = extract_entities_demo(docs[other_doc_id], doc_filenames.get(other_doc_id, ""))
                other_entity_set = set()
                for entities in other_entities.values():
                    other_entity_set.update(entities)
                
                if not other_entity_set:
                    continue
                
                # Calculate Jaccard similarity
                intersection = len(current_entity_set.intersection(other_entity_set))
                union = len(current_entity_set.union(other_entity_set))
                
                if union > 0:
                    similarity = intersection / union
                    
                    if similarity >= similarity_threshold:
                        # Determine relationship type
                        rel_type = "related"
                        if intersection >= 3:
                            rel_type = "strongly_related"
                        elif intersection >= 2:
                            rel_type = "moderately_related"
                        
                        related.append((other_doc_id, similarity, rel_type))
            
            # Sort by similarity
            related.sort(key=lambda x: x[1], reverse=True)
            return related[:5]  # Top 5 related documents
            
        except Exception as e:
            print(f"Relationship analysis failed: {e}")
            return []
    
    def get_entity_network(self) -> Dict[str, Any]:
        """Get the entity co-occurrence network."""
        with self.lock:
            # Convert to network format
            nodes = set()
            edges = []
            
            for entity, related_entities in self.entity_co_occurrences.items():
                nodes.add(entity)
                for related_entity, count in related_entities.items():
                    if count > 1:  # Only include relationships seen multiple times
                        nodes.add(related_entity)
                        edges.append({
                            "source": entity,
                            "target": related_entity,
                            "weight": count
                        })
            
            return {
                "nodes": list(nodes),
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
    
    def analyze_entity_clusters(self) -> Dict[str, Any]:
        """Analyze entity clusters and communities."""
        with self.lock:
            network = self.get_entity_network()
            
            # Simple clustering based on connection strength
            clusters = {}
            cluster_id = 0
            processed_nodes = set()
            
            for edge in network["edges"]:
                source = edge["source"]
                target = edge["target"]
                weight = edge["weight"]
                
                if source not in processed_nodes and target not in processed_nodes:
                    # Create new cluster
                    if weight > 2:  # Strong connection threshold
                        clusters[f"cluster_{cluster_id}"] = {
                            "entities": [source, target],
                            "strength": weight,
                            "theme": self._infer_cluster_theme([source, target])
                        }
                        processed_nodes.update([source, target])
                        cluster_id += 1
            
            return {
                "clusters": clusters,
                "cluster_count": len(clusters),
                "clustered_entities": len(processed_nodes),
                "total_entities": network["node_count"]
            }
    
    def _infer_cluster_theme(self, entities: List[str]) -> str:
        """Infer the theme of an entity cluster."""
        types = [entity.split(":")[0] for entity in entities if ":" in entity]
        
        if "equipment" in types and "incident" in types:
            return "equipment_incidents"
        elif "equipment" in types and "location" in types:
            return "equipment_locations"
        elif all(t == "equipment" for t in types):
            return "equipment_group"
        elif all(t == "location" for t in types):
            return "location_group"
        else:
            return "mixed_entities"
    
    def get_relationship_insights(self, docs: Dict, doc_filenames: Dict) -> Dict[str, Any]:
        """Generate insights about document relationships."""
        with self.lock:
            total_docs = len(docs)
            
            if total_docs < 2:
                return {"message": "Need at least 2 documents for relationship analysis"}
            
            # Find highly connected documents
            connection_counts = defaultdict(int)
            
            for doc_id in docs.keys():
                related = self.find_related_capsules(doc_id, docs, doc_filenames)
                connection_counts[doc_id] = len(related)
            
            # Identify hubs (highly connected documents)
            avg_connections = sum(connection_counts.values()) / len(connection_counts)
            hubs = [
                doc_id for doc_id, count in connection_counts.items()
                if count > avg_connections * 1.5
            ]
            
            # Identify isolated documents
            isolated = [
                doc_id for doc_id, count in connection_counts.items()
                if count == 0
            ]
            
            return {
                "total_documents": total_docs,
                "average_connections": avg_connections,
                "hub_documents": len(hubs),
                "isolated_documents": len(isolated),
                "most_connected": max(connection_counts.items(), key=lambda x: x[1], default=(None, 0)),
                "connection_distribution": dict(connection_counts)
            }


# Global instance
relationship_miner = RelationshipMiner()
