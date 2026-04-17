"""
Graph Database Manager for Neo4j
Handles graph embeddings, entity relationships, and complex query processing
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase # type: ignore
import networkx as nx # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np # type: ignore
from datetime import datetime

class GraphDatabaseManager:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        """
        Initialize Neo4j connection and embedding model
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.is_connected = False
        
        try:
            self.connect()
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            print("Graph functionality will use in-memory NetworkX as fallback")
            self.fallback_graph = nx.MultiDiGraph()
            self.node_embeddings = {}
    
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("MATCH () RETURN count(*) as count")
            self.is_connected = True
            print("Successfully connected to Neo4j")
            self._create_indexes()
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.is_connected = False
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def _create_indexes(self):
        """Create necessary indexes for better performance"""
        if not self.is_connected:
            return
            
        indexes = [
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX document_id_idx IF NOT EXISTS FOR (d:Document) ON (d.document_id)",
            "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    print(f"Index creation warning: {e}")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        if self.is_connected:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        else:
            self.fallback_graph.clear()
            self.node_embeddings.clear()
        print("Database cleared")
    
    def create_document_node(self, document_id: str, title: str, content: str, metadata: Dict = None):
        """Create a document node with embeddings"""
        embedding = self.embedding_model.encode(content).tolist()
        
        if self.is_connected:
            with self.driver.session() as session:
                session.run("""
                    CREATE (d:Document {
                        document_id: $document_id,
                        title: $title,
                        content: $content,
                        embedding: $embedding,
                        metadata: $metadata,
                        created_at: datetime()
                    })
                """, document_id=document_id, title=title, content=content, 
                    embedding=embedding, metadata=metadata or {})
        else:
            self.fallback_graph.add_node(document_id, 
                                       type="Document",
                                       title=title,
                                       content=content,
                                       metadata=metadata or {})
            self.node_embeddings[document_id] = embedding
        
        return document_id
    
    def create_entity_node(self, entity_name: str, entity_type: str, description: str = "", 
                          document_id: str = None, properties: Dict = None):
        """Create an entity node with embeddings"""
        embedding_text = f"{entity_name} {entity_type} {description}"
        embedding = self.embedding_model.encode(embedding_text).tolist()
        
        entity_id = f"entity_{hashlib.md5(entity_name.encode()).hexdigest()[:8]}"
        
        if self.is_connected:
            with self.driver.session() as session:
                session.run("""
                    MERGE (e:Entity {name: $entity_name})
                    SET e.entity_id = $entity_id,
                        e.type = $entity_type,
                        e.description = $description,
                        e.embedding = $embedding,
                        e.properties = $properties,
                        e.updated_at = datetime()
                """, entity_name=entity_name, entity_id=entity_id, entity_type=entity_type,
                    description=description, embedding=embedding, properties=properties or {})
                
                # Link to document if provided
                if document_id:
                    session.run("""
                        MATCH (e:Entity {entity_id: $entity_id})
                        MATCH (d:Document {document_id: $document_id})
                        MERGE (e)-[:MENTIONED_IN]->(d)
                    """, entity_id=entity_id, document_id=document_id)
        else:
            self.fallback_graph.add_node(entity_id,
                                       type="Entity",
                                       name=entity_name,
                                       entity_type=entity_type,
                                       description=description,
                                       properties=properties or {})
            self.node_embeddings[entity_id] = embedding
            
            if document_id:
                self.fallback_graph.add_edge(entity_id, document_id, type="MENTIONED_IN")
        
        return entity_id
    
    def create_concept_node(self, concept_name: str, definition: str, category: str = "",
                           document_id: str = None):
        """Create a concept node with embeddings"""
        embedding_text = f"{concept_name} {definition} {category}"
        embedding = self.embedding_model.encode(embedding_text).tolist()
        
        concept_id = f"concept_{hashlib.md5(concept_name.encode()).hexdigest()[:8]}"
        
        if self.is_connected:
            with self.driver.session() as session:
                session.run("""
                    MERGE (c:Concept {name: $concept_name})
                    SET c.concept_id = $concept_id,
                        c.definition = $definition,
                        c.category = $category,
                        c.embedding = $embedding,
                        c.updated_at = datetime()
                """, concept_name=concept_name, concept_id=concept_id, 
                    definition=definition, category=category, embedding=embedding)
                
                if document_id:
                    session.run("""
                        MATCH (c:Concept {concept_id: $concept_id})
                        MATCH (d:Document {document_id: $document_id})
                        MERGE (c)-[:DEFINED_IN]->(d)
                    """, concept_id=concept_id, document_id=document_id)
        else:
            self.fallback_graph.add_node(concept_id,
                                       type="Concept",
                                       name=concept_name,
                                       definition=definition,
                                       category=category)
            self.node_embeddings[concept_id] = embedding
            
            if document_id:
                self.fallback_graph.add_edge(concept_id, document_id, type="DEFINED_IN")
        
        return concept_id
    
    def create_relationship(self, from_node_id: str, to_node_id: str, relationship_type: str,
                           properties: Dict = None, strength: float = 1.0):
        """Create a relationship between nodes"""
        if self.is_connected:
            with self.driver.session() as session:
                session.run(f"""
                    MATCH (a) WHERE a.entity_id = $from_id OR a.document_id = $from_id OR a.concept_id = $from_id
                    MATCH (b) WHERE b.entity_id = $to_id OR b.document_id = $to_id OR b.concept_id = $to_id
                    MERGE (a)-[r:{relationship_type}]->(b)
                    SET r.properties = $properties,
                        r.strength = $strength,
                        r.created_at = datetime()
                """, from_id=from_node_id, to_id=to_node_id, 
                    properties=properties or {}, strength=strength)
        else:
            self.fallback_graph.add_edge(from_node_id, to_node_id,
                                       type=relationship_type,
                                       properties=properties or {},
                                       strength=strength)
    
    def find_similar_nodes(self, query_text: str, node_type: str = None, limit: int = 5):
        """Find nodes similar to query using embeddings"""
        query_embedding = self.embedding_model.encode(query_text)
        
        if self.is_connected:
            # Use Neo4j vector similarity (if available)
            with self.driver.session() as session:
                cypher_query = """
                    MATCH (n)
                    WHERE ($node_type IS NULL OR n:""" + (node_type or "Entity") + """)
                    AND n.embedding IS NOT NULL
                    RETURN n, n.embedding as embedding
                    LIMIT 100
                """
                results = session.run(cypher_query, node_type=node_type)
                
                similarities = []
                for record in results:
                    node = record["n"]
                    node_embedding = np.array(record["embedding"])
                    similarity = np.dot(query_embedding, node_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
                    )
                    similarities.append((similarity, dict(node)))
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x[0], reverse=True)
                return [(score, node) for score, node in similarities[:limit]]
        else:
            # Fallback to NetworkX
            similarities = []
            for node_id, embedding in self.node_embeddings.items():
                if node_type:
                    node_data = self.fallback_graph.nodes[node_id]
                    if node_data.get('type') != node_type:
                        continue
                
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((similarity, self.fallback_graph.nodes[node_id]))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            return similarities[:limit]
    
    def find_relationships(self, entity_name: str, relationship_types: List[str] = None,
                          max_depth: int = 2):
        """Find relationships involving an entity"""
        if self.is_connected:
            with self.driver.session() as session:
                # Build dynamic Cypher query
                rel_filter = ""
                if relationship_types:
                    rel_types = "|".join(relationship_types)
                    rel_filter = f":{rel_types}"
                
                cypher_query = f"""
                    MATCH path = (start)-[r{rel_filter}*1..{max_depth}]-(connected)
                    WHERE start.name CONTAINS $entity_name OR start.entity_id CONTAINS $entity_name
                    RETURN path, relationships(path) as rels, nodes(path) as nodes
                    LIMIT 50
                """
                
                results = session.run(cypher_query, entity_name=entity_name)
                
                paths = []
                for record in results:
                    path_nodes = [dict(node) for node in record["nodes"]]
                    path_rels = [dict(rel) for rel in record["rels"]]
                    paths.append({
                        'nodes': path_nodes,
                        'relationships': path_rels
                    })
                
                return paths
        else:
            # Fallback using NetworkX
            paths = []
            for node_id, node_data in self.fallback_graph.nodes(data=True):
                if entity_name.lower() in node_data.get('name', '').lower():
                    # Find paths from this node
                    for target in self.fallback_graph.nodes():
                        if target != node_id:
                            try:
                                if nx.has_path(self.fallback_graph, node_id, target):
                                    path = nx.shortest_path(self.fallback_graph, node_id, target)
                                    if len(path) <= max_depth + 1:
                                        path_data = {
                                            'nodes': [self.fallback_graph.nodes[n] for n in path],
                                            'relationships': []
                                        }
                                        paths.append(path_data)
                            except:
                                continue
                    break
            
            return paths[:50]  # Limit results
    
    def execute_graph_query(self, query: str, parameters: Dict = None):
        """Execute custom Cypher query"""
        if not self.is_connected:
            return {"error": "Neo4j not connected. Using fallback graph storage."}
        
        try:
            with self.driver.session() as session:
                results = session.run(query, parameters or {})
                return [record.data() for record in results]
        except Exception as e:
            return {"error": str(e)}
    
    def get_graph_statistics(self):
        """Get statistics about the graph database"""
        if self.is_connected:
            with self.driver.session() as session:
                stats = {}
                
                # Count nodes by type
                result = session.run("""
                    MATCH (n) 
                    RETURN labels(n) as labels, count(n) as count
                """)
                node_counts = {}
                for record in result:
                    labels = record["labels"]
                    count = record["count"]
                    label_key = ":".join(labels) if labels else "Unlabeled"
                    node_counts[label_key] = count
                
                # Count relationships
                result = session.run("""
                    MATCH ()-[r]->() 
                    RETURN type(r) as rel_type, count(r) as count
                """)
                rel_counts = {}
                for record in result:
                    rel_counts[record["rel_type"]] = record["count"]
                
                stats = {
                    "nodes": node_counts,
                    "relationships": rel_counts,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(rel_counts.values())
                }
                
                return stats
        else:
            return {
                "nodes": {"NetworkX_Fallback": len(self.fallback_graph.nodes)},
                "relationships": {"NetworkX_Edges": len(self.fallback_graph.edges)},
                "total_nodes": len(self.fallback_graph.nodes),
                "total_relationships": len(self.fallback_graph.edges)
            }
    
    def generate_cypher_from_natural_language(self, natural_query: str):
        """
        Generate Cypher query from natural language
        This is a simple template-based approach - can be enhanced with LLM
        """
        query_lower = natural_query.lower()
        
        # Template patterns
        if "relationship" in query_lower or "relation" in query_lower:
            if "between" in query_lower:
                return """
                MATCH (a)-[r]-(b)
                WHERE a.name CONTAINS $entity1 AND b.name CONTAINS $entity2
                RETURN a, r, b
                LIMIT 10
                """
        
        elif "connect" in query_lower or "link" in query_lower:
            return """
            MATCH path = (start)-[*1..3]-(end)
            WHERE start.name CONTAINS $entity OR end.name CONTAINS $entity
            RETURN path
            LIMIT 20
            """
        
        elif "similar" in query_lower or "related" in query_lower:
            return """
            MATCH (n)-[r]-(connected)
            WHERE n.name CONTAINS $entity
            RETURN n, r, connected
            LIMIT 15
            """
        
        else:
            # Default: find entities matching the query
            return """
            MATCH (n)
            WHERE n.name CONTAINS $query OR n.description CONTAINS $query
            RETURN n
            LIMIT 10
            """

# Example usage and testing
if __name__ == "__main__":
    # Test with fallback mode (no Neo4j required)
    graph_db = GraphDatabaseManager()
    
    # Create some test data
    doc_id = graph_db.create_document_node("doc1", "Machine Learning Basics", 
                                         "Machine learning is a subset of artificial intelligence...")
    
    entity1 = graph_db.create_entity_node("Machine Learning", "Concept", 
                                        "A method of data analysis", doc_id)
    entity2 = graph_db.create_entity_node("Artificial Intelligence", "Field", 
                                        "Intelligence demonstrated by machines", doc_id)
    
    # Create relationship
    graph_db.create_relationship(entity1, entity2, "SUBSET_OF", strength=0.9)
    
    # Test similarity search
    results = graph_db.find_similar_nodes("artificial intelligence")
    print("Similar nodes:", results)
    
    # Test statistics
    stats = graph_db.get_graph_statistics()
    print("Graph statistics:", stats)