"""
Graph Query Processor
Handles complex queries using graph embeddings and generates appropriate responses
"""

import re
from typing import Dict, List, Any, Tuple
import json

class GraphQueryProcessor:
    def __init__(self, graph_db_manager):
        self.graph_db = graph_db_manager
        self.query_templates = self._initialize_query_templates()
        self.response_formatter = ResponseFormatter()
    
    def _initialize_query_templates(self):
        """Initialize Cypher query templates for different types of questions"""
        return {
            'relationship': {
                'patterns': ['relationship', 'relation', 'connect', 'link', 'between'],
                'cypher': """
                MATCH (a)-[r]-(b)
                WHERE (a.name CONTAINS $entity1 OR a.description CONTAINS $entity1)
                AND (b.name CONTAINS $entity2 OR b.description CONTAINS $entity2)
                RETURN a, r, b, type(r) as relationship_type
                LIMIT 10
                """
            },
            'comparison': {
                'patterns': ['compare', 'difference', 'versus', 'vs', 'similar', 'different'],
                'cypher': """
                MATCH (a {name: $entity1})-[r1]-(common)-[r2]-(b {name: $entity2})
                RETURN a, b, common, type(r1) as rel1_type, type(r2) as rel2_type
                UNION
                MATCH (a) WHERE a.name CONTAINS $entity1
                MATCH (b) WHERE b.name CONTAINS $entity2
                RETURN a, b, null as common, null as rel1_type, null as rel2_type
                LIMIT 15
                """
            },
            'hierarchy': {
                'patterns': ['hierarchy', 'structure', 'levels', 'categories', 'classification'],
                'cypher': """
                MATCH path = (root)-[:INCLUDES|SUBSET_OF*1..3]->(leaf)
                WHERE root.name CONTAINS $entity OR leaf.name CONTAINS $entity
                RETURN path, nodes(path) as hierarchy_nodes, relationships(path) as hierarchy_rels
                LIMIT 20
                """
            },
            'causation': {
                'patterns': ['cause', 'effect', 'leads to', 'results in', 'because', 'due to'],
                'cypher': """
                MATCH (cause)-[:CAUSES|LEADS_TO|RESULTS_IN]->(effect)
                WHERE cause.name CONTAINS $entity OR effect.name CONTAINS $entity
                RETURN cause, effect, 'causation' as relationship_type
                UNION
                MATCH (effect)-[:CAUSED_BY|RESULTS_FROM]->(cause)
                WHERE cause.name CONTAINS $entity OR effect.name CONTAINS $entity
                RETURN cause, effect, 'causation' as relationship_type
                LIMIT 10
                """
            },
            'definition': {
                'patterns': ['what is', 'define', 'definition', 'meaning', 'explain'],
                'cypher': """
                MATCH (entity)
                WHERE entity.name CONTAINS $entity
                OPTIONAL MATCH (entity)-[:DEFINED_AS|MEANS]->(definition)
                OPTIONAL MATCH (entity)-[:SUBSET_OF]->(parent)
                OPTIONAL MATCH (child)-[:SUBSET_OF]->(entity)
                RETURN entity, definition, parent, collect(child) as children
                LIMIT 5
                """
            },
            'association': {
                'patterns': ['related', 'associated', 'connected', 'linked'],
                'cypher': """
                MATCH (center)-[r]-(associated)
                WHERE center.name CONTAINS $entity
                RETURN center, associated, type(r) as relationship_type, r.strength as strength
                ORDER BY r.strength DESC
                LIMIT 20
                """
            },
            'find_similar': {
                'patterns': ['similar to', 'like', 'resembles', 'comparable'],
                'cypher': """
                MATCH (target) WHERE target.name CONTAINS $entity
                MATCH (similar)-[:SIMILAR_TO|RELATED_TO]-(target)
                RETURN target, similar, 'similarity' as relationship_type
                UNION
                MATCH (node)
                WHERE node.name CONTAINS $entity AND node.embedding IS NOT NULL
                RETURN node, null as similar, 'embedding_match' as relationship_type
                LIMIT 15
                """
            }
        }
    
    def process_complex_query(self, query: str, entities: List[str] = None) -> Dict[str, Any]:
        """
        Process complex query using graph database
        """
        try:
            # Determine query type and extract parameters
            query_type, parameters = self._analyze_query(query, entities)
            
            # Execute appropriate Cypher query
            if query_type in self.query_templates:
                cypher_query = self.query_templates[query_type]['cypher']
                results = self._execute_cypher_query(cypher_query, parameters)
            else:
                # Fallback to general search
                results = self._general_graph_search(query, entities)
            
            # Format response
            formatted_response = self.response_formatter.format_graph_response(
                query, query_type, results
            )
            
            return {
                'success': True,
                'query_type': query_type,
                'parameters': parameters,
                'raw_results': results,
                'formatted_response': formatted_response,
                'source': 'graph_database'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fallback_response': f"I encountered an error processing your complex query: {e}",
                'source': 'graph_database'
            }
    
    def _analyze_query(self, query: str, entities: List[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Analyze query to determine type and extract parameters"""
        query_lower = query.lower()
        
        # Extract entities from query if not provided
        if not entities:
            entities = self._extract_query_entities(query)
        
        # Determine query type based on patterns
        for query_type, template in self.query_templates.items():
            for pattern in template['patterns']:
                if pattern in query_lower:
                    parameters = self._extract_parameters(query, query_type, entities)
                    return query_type, parameters
        
        # Default to association type
        parameters = self._extract_parameters(query, 'association', entities)
        return 'association', parameters
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entities from query text"""
        # Simple entity extraction based on capitalized words and common patterns
        entities = []
        
        # Find quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_terms)
        
        # Find capitalized terms (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(capitalized)
        
        # Find technical terms
        technical_terms = re.findall(r'\b\w+(?:ing|tion|ment|ism|ity)\b', query)
        entities.extend(technical_terms)
        
        # Remove common words
        stop_words = {'The', 'This', 'That', 'When', 'Where', 'How', 'Why', 'What', 'Which'}
        entities = [e for e in entities if e not in stop_words and len(e) > 2]
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_parameters(self, query: str, query_type: str, entities: List[str]) -> Dict[str, Any]:
        """Extract parameters needed for Cypher queries"""
        parameters = {}
        
        if entities:
            if len(entities) >= 2 and query_type in ['relationship', 'comparison']:
                parameters['entity1'] = entities[0]
                parameters['entity2'] = entities[1]
            else:
                parameters['entity'] = entities[0]
        else:
            # Extract from query patterns
            if 'between' in query.lower():
                # Try to find entities around 'between'
                between_pattern = r'between\s+(.+?)\s+and\s+(.+?)(?:\s|$|\.|\?)'
                match = re.search(between_pattern, query, re.IGNORECASE)
                if match:
                    parameters['entity1'] = match.group(1).strip()
                    parameters['entity2'] = match.group(2).strip()
            
            if not parameters and entities:
                parameters['entity'] = entities[0]
            elif not parameters:
                # Use the longest word as entity
                words = query.split()
                longest_word = max(words, key=len) if words else 'unknown'
                parameters['entity'] = longest_word
        
        return parameters
    
    def _execute_cypher_query(self, cypher_query: str, parameters: Dict[str, Any]) -> List[Dict]:
        """Execute Cypher query using the graph database manager"""
        try:
            results = self.graph_db.execute_graph_query(cypher_query, parameters)
            return results if results and not isinstance(results, dict) else []
        except Exception as e:
            print(f"Cypher query execution error: {e}")
            return []
    
    def _general_graph_search(self, query: str, entities: List[str] = None) -> List[Dict]:
        """Perform general graph search when no specific pattern matches"""
        results = []
        
        try:
            # Search for similar nodes
            if entities:
                for entity in entities[:3]:  # Limit to first 3 entities
                    similar_nodes = self.graph_db.find_similar_nodes(entity, limit=5)
                    for score, node in similar_nodes:
                        results.append({
                            'type': 'similar_node',
                            'entity': entity,
                            'node': node,
                            'similarity_score': score
                        })
            
            # Search for relationships
            if entities:
                relationships = self.graph_db.find_relationships(entities[0])
                for rel_path in relationships[:10]:  # Limit results
                    results.append({
                        'type': 'relationship_path',
                        'path': rel_path
                    })
        
        except Exception as e:
            print(f"General graph search error: {e}")
        
        return results

class ResponseFormatter:
    """Formats graph database responses into human-readable text"""
    
    def format_graph_response(self, query: str, query_type: str, results: List[Dict]) -> str:
        """Format graph database results into a coherent response"""
        
        if not results:
            return self._generate_no_results_response(query, query_type)
        
        if query_type == 'relationship':
            return self._format_relationship_response(results)
        elif query_type == 'comparison':
            return self._format_comparison_response(results)
        elif query_type == 'hierarchy':
            return self._format_hierarchy_response(results)
        elif query_type == 'causation':
            return self._format_causation_response(results)
        elif query_type == 'definition':
            return self._format_definition_response(results)
        elif query_type == 'association':
            return self._format_association_response(results)
        else:
            return self._format_general_response(results)
    
    def _generate_no_results_response(self, query: str, query_type: str) -> str:
        """Generate response when no results are found"""
        return f"""I couldn't find specific information in the graph database for your {query_type} query about: "{query}"

This might be because:
- The entities mentioned are not in the knowledge base
- The relationships haven't been established yet
- The query might need to be reformulated

Try asking about more general concepts or check if the documents have been properly processed."""
    
    def _format_relationship_response(self, results: List[Dict]) -> str:
        """Format relationship query results"""
        response = "**Relationships Found:**\n\n"
        
        relationships = {}
        for result in results:
            if 'a' in result and 'b' in result and 'relationship_type' in result:
                rel_type = result['relationship_type']
                entity_a = result['a'].get('name', 'Unknown')
                entity_b = result['b'].get('name', 'Unknown')
                
                if rel_type not in relationships:
                    relationships[rel_type] = []
                relationships[rel_type].append((entity_a, entity_b))
        
        for rel_type, pairs in relationships.items():
            response += f"**{rel_type.replace('_', ' ').title()}:**\n"
            for entity_a, entity_b in pairs[:5]:  # Limit to 5 per type
                response += f"• {entity_a} → {entity_b}\n"
            response += "\n"
        
        if not relationships:
            response = "No direct relationships found in the graph database."
        
        return response
    
    def _format_comparison_response(self, results: List[Dict]) -> str:
        """Format comparison query results"""
        response = "**Comparison Analysis:**\n\n"
        
        entities = set()
        common_connections = []
        
        for result in results:
            if 'a' in result and 'b' in result:
                entities.add(result['a'].get('name', 'Unknown'))
                entities.add(result['b'].get('name', 'Unknown'))
                
                if result.get('common'):
                    common_connections.append({
                        'entity_a': result['a'].get('name'),
                        'entity_b': result['b'].get('name'),
                        'common': result['common'].get('name'),
                        'relation_a': result.get('rel1_type'),
                        'relation_b': result.get('rel2_type')
                    })
        
        if len(entities) >= 2:
            entity_list = list(entities)
            response += f"Comparing **{entity_list[0]}** and **{entity_list[1]}**:\n\n"
            
            if common_connections:
                response += "**Common Connections:**\n"
                for conn in common_connections[:5]:
                    response += f"• Both connect to **{conn['common']}**\n"
                    if conn['relation_a']:
                        response += f"  - {conn['entity_a']} via {conn['relation_a']}\n"
                    if conn['relation_b']:
                        response += f"  - {conn['entity_b']} via {conn['relation_b']}\n"
                response += "\n"
            else:
                response += "No common connections found in the current knowledge base.\n"
        
        return response
    
    def _format_hierarchy_response(self, results: List[Dict]) -> str:
        """Format hierarchy query results"""
        response = "**Hierarchical Structure:**\n\n"
        
        for result in results:
            if 'hierarchy_nodes' in result:
                nodes = result['hierarchy_nodes']
                if len(nodes) > 1:
                    hierarchy_path = " → ".join([node.get('name', 'Unknown') for node in nodes])
                    response += f"• {hierarchy_path}\n"
        
        if response == "**Hierarchical Structure:**\n\n":
            response = "No clear hierarchical relationships found in the graph database."
        
        return response
    
    def _format_causation_response(self, results: List[Dict]) -> str:
        """Format causation query results"""
        response = "**Cause-Effect Relationships:**\n\n"
        
        for result in results:
            if 'cause' in result and 'effect' in result:
                cause_name = result['cause'].get('name', 'Unknown')
                effect_name = result['effect'].get('name', 'Unknown')
                response += f"• **{cause_name}** causes **{effect_name}**\n"
        
        if response == "**Cause-Effect Relationships:**\n\n":
            response = "No causal relationships found in the graph database."
        
        return response
    
    def _format_definition_response(self, results: List[Dict]) -> str:
        """Format definition query results"""
        response = ""
        
        for result in results:
            if 'entity' in result:
                entity = result['entity']
                entity_name = entity.get('name', 'Unknown')
                
                response += f"**{entity_name}**\n"
                
                if entity.get('description'):
                    response += f"*Description:* {entity['description']}\n"
                
                if entity.get('definition'):
                    response += f"*Definition:* {entity['definition']}\n"
                
                if result.get('parent'):
                    parent_name = result['parent'].get('name', 'Unknown')
                    response += f"*Category:* Subset of {parent_name}\n"
                
                if result.get('children'):
                    children_names = [child.get('name', 'Unknown') for child in result['children']]
                    if children_names:
                        response += f"*Includes:* {', '.join(children_names)}\n"
                
                response += "\n"
                break  # Only show first result for definition queries
        
        if not response:
            response = "No definition found in the graph database."
        
        return response
    
    def _format_association_response(self, results: List[Dict]) -> str:
        """Format association query results"""
        response = "**Associated Concepts:**\n\n"
        
        associations = {}
        for result in results:
            if 'center' in result and 'associated' in result:
                rel_type = result.get('relationship_type', 'RELATED_TO')
                strength = result.get('strength', 'Unknown')
                
                center_name = result['center'].get('name', 'Unknown')
                assoc_name = result['associated'].get('name', 'Unknown')
                
                if rel_type not in associations:
                    associations[rel_type] = []
                associations[rel_type].append((center_name, assoc_name, strength))
        
        for rel_type, items in associations.items():
            response += f"**{rel_type.replace('_', ' ').title()}:**\n"
            for center, assoc, strength in items[:5]:
                strength_str = f" (strength: {strength})" if strength != 'Unknown' else ""
                response += f"• {center} ↔ {assoc}{strength_str}\n"
            response += "\n"
        
        if not associations:
            response = "No associations found in the graph database."
        
        return response
    
    def _format_general_response(self, results: List[Dict]) -> str:
        """Format general search results"""
        response = "**Search Results:**\n\n"
        
        for result in results:
            if result.get('type') == 'similar_node':
                entity = result.get('entity', 'Unknown')
                node = result.get('node', {})
                score = result.get('similarity_score', 0)
                
                response += f"• **{node.get('name', 'Unknown')}** (similarity: {score:.2f})\n"
                if node.get('description'):
                    response += f"  {node['description'][:100]}...\n"
            
            elif result.get('type') == 'relationship_path':
                path = result.get('path', {})
                nodes = path.get('nodes', [])
                if len(nodes) > 1:
                    path_str = " → ".join([node.get('name', 'Unknown') for node in nodes])
                    response += f"• Path: {path_str}\n"
        
        if response == "**Search Results:**\n\n":
            response = "No relevant information found in the graph database."
        
        return response

# Example usage
if __name__ == "__main__":
    # This would be used with an actual GraphDatabaseManager instance
    print("Graph Query Processor initialized")
    print("Use with GraphDatabaseManager instance for processing complex queries")