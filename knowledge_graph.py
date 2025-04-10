"""
Knowledge Graph implementation for the AI Research System
"""

from advanced_ai_analyzer import *
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pickle
import os
from tqdm import tqdm
import community as community_louvain
import scipy.sparse

class KnowledgeGraph:
    """A graph representation of concepts and their relationships in the knowledge base"""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.graph = nx.Graph()
        self.concept_embeddings = {}
        self.community_map = {}
        self.concept_importance = {}
        self.relation_types = ['co_occurrence', 'semantic_similarity', 'citation_link']
        self.adj_matrix = None
        self.concept_index = {}
        
        # Create graph directory
        self.graph_dir = os.path.join(CONFIG['models_dir'], 'knowledge_graph')
        os.makedirs(self.graph_dir, exist_ok=True)
        
    def build_graph(self, papers):
        """Build graph incrementally with sparse representations"""
        logger.info("Building knowledge graph from knowledge base")
        
        # Use sparse matrices for large graphs
        self.adj_matrix = scipy.sparse.lil_matrix((len(self.kb.concepts), len(self.kb.concepts)))
        
        # Create concept index
        self.concept_index = {concept: i for i, concept in enumerate(self.kb.concepts)}
        
        # Process papers in chunks
        for i in range(0, len(papers), 1000):
            chunk = papers[i:i+1000]
            for paper in chunk:
                # Only store top 20 concepts per paper to reduce edges
                concepts = sorted(paper['concepts'], key=lambda x: x['score'], reverse=True)[:20]
                
                # Add concepts as nodes
                for concept in concepts:
                    if concept not in self.graph:
                        self.graph.add_node(concept, weight=1)
                    else:
                        self.graph.nodes[concept]['weight'] += 1
                
                # Track concept co-occurrence
                co_occurrence = defaultdict(Counter)
                
                # Build co-occurrence edges
                for j, concept1 in enumerate(concepts):
                    for concept2 in concepts[j+1:]:
                        co_occurrence[concept1][concept2] += 1
                        co_occurrence[concept2][concept1] += 1
                
                # Add edges with weights based on co-occurrence
                for concept1, related in co_occurrence.items():
                    for concept2, weight in related.items():
                        if weight >= 2:  # Only add edges with multiple co-occurrences
                            if self.graph.has_edge(concept1, concept2):
                                self.graph[concept1][concept2]['weight'] += weight
                            else:
                                self.graph.add_edge(concept1, concept2, 
                                                  weight=weight,
                                                  relation='co_occurrence')
            
            # Periodically prune weak connections
            if i % 5000 == 0:
                self._prune_weak_edges(threshold=0.1)
        
        # Add semantic similarity edges if we have concept embeddings
        if hasattr(self.kb, 'concept_embeddings') and self.kb.concept_embeddings:
            logger.info("Adding semantic similarity edges based on concept embeddings")
            concept_embeddings = {}
            
            # Get embeddings for filtered concepts
            for concept in self.kb.concepts:
                if concept in self.kb.concept_embeddings:
                    concept_embeddings[concept] = self.kb.concept_embeddings[concept]
            
            # Calculate cosine similarity between concepts
            similarity_edges = 0
            for concept1 in tqdm(concept_embeddings.keys(), desc="Calculating similarities"):
                emb1 = concept_embeddings[concept1]
                
                # Find top 5 similar concepts
                similarities = []
                for concept2, emb2 in concept_embeddings.items():
                    if concept1 != concept2:
                        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        similarities.append((concept2, sim))
                
                # Add edges for top similar concepts
                for concept2, sim in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]:
                    if sim > 0.7:  # Only add high similarity edges
                        if self.graph.has_edge(concept1, concept2):
                            self.graph[concept1][concept2]['weight'] += sim
                        else:
                            self.graph.add_edge(concept1, concept2, 
                                              weight=sim,
                                              relation='semantic_similarity')
                        similarity_edges += 1
            
            logger.info(f"Added {similarity_edges} semantic similarity edges to knowledge graph")
        
        # Add citation link edges from paper citations
        if hasattr(self.kb, 'citation_network') and self.kb.citation_network:
            logger.info("Adding citation link edges based on citation network")
            citation_edges = 0
            
            # Track which concepts are mentioned in which papers
            concept_to_papers = defaultdict(set)
            for paper_id, paper in self.kb.papers.items():
                if 'concepts' in paper and paper['concepts']:
                    for concept in paper['concepts']:
                        if concept in self.kb.concepts:
                            concept_to_papers[concept].add(paper_id)
            
            # Connect concepts in papers that cite each other
            for paper_id, paper in tqdm(self.kb.papers.items(), desc="Processing citations"):
                if 'citations' in paper and paper['citations']:
                    # Get concepts in this paper
                    source_concepts = {c for c in paper.get('concepts', []) if c in self.kb.concepts}
                    
                    # For each cited paper
                    for cited_id in paper['citations']:
                        if cited_id in self.kb.papers:
                            # Get concepts in cited paper
                            cited_concepts = {c for c in self.kb.papers[cited_id].get('concepts', []) 
                                            if c in self.kb.concepts}
                            
                            # Connect concepts across citing-cited papers
                            for src_concept in source_concepts:
                                for dst_concept in cited_concepts:
                                    if src_concept != dst_concept:
                                        # Add edge or increase weight if exists
                                        if self.graph.has_edge(src_concept, dst_concept):
                                            self.graph[src_concept][dst_concept]['weight'] += 1
                                        else:
                                            self.graph.add_edge(src_concept, dst_concept,
                                                              weight=1,
                                                              relation='citation_link')
                                            citation_edges += 1
            
            logger.info(f"Added {citation_edges} citation link edges to knowledge graph")
        
        # Calculate node centrality to identify important concepts
        logger.info("Calculating concept importance using PageRank")
        self.concept_importance = nx.pagerank(self.graph, weight='weight')
        
        # Identify communities using Louvain method
        logger.info("Identifying concept communities in knowledge graph")
        self.community_map = community_louvain.best_partition(self.graph, weight='weight')
        
        # Count communities
        community_counts = Counter(self.community_map.values())
        logger.info(f"Identified {len(community_counts)} communities in knowledge graph")
        
        # Save graph
        self.save_graph()
        
        return True
    
    def _prune_weak_edges(self, threshold=0.1):
        """Prune weak edges from the graph"""
        logger.info("Pruning weak edges from knowledge graph")
        
        # Remove edges with weights below threshold
        weak_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['weight'] < threshold]
        self.graph.remove_edges_from(weak_edges)
        
        logger.info(f"Removed {len(weak_edges)} weak edges from knowledge graph")
    
    def get_concept_communities(self, top_n=10):
        """Get the most important concepts in each community"""
        if not self.community_map or not self.concept_importance:
            logger.warning("Knowledge graph not built or no communities identified")
            return []
        
        # Group concepts by community
        community_concepts = defaultdict(list)
        for concept, community_id in self.community_map.items():
            importance = self.concept_importance.get(concept, 0)
            community_concepts[community_id].append((concept, importance))
        
        # Get top concepts in each community
        top_communities = []
        for community_id, concepts in community_concepts.items():
            # Sort by importance
            sorted_concepts = sorted(concepts, key=lambda x: x[1], reverse=True)
            
            # Get top concepts
            top_concepts = [c[0] for c in sorted_concepts[:top_n]]
            
            # Only include communities with at least 3 concepts
            if len(top_concepts) >= 3:
                community_size = len(concepts)
                avg_importance = sum(c[1] for c in concepts) / max(1, community_size)
                
                top_communities.append({
                    'id': community_id,
                    'size': community_size,
                    'importance': avg_importance,
                    'concepts': top_concepts
                })
        
        # Sort communities by importance
        sorted_communities = sorted(top_communities, key=lambda x: x['importance'], reverse=True)
        
        return sorted_communities
    
    def get_related_concepts(self, concept, relation_type=None, limit=10):
        """Get concepts related to the given concept"""
        if not self.graph.has_node(concept):
            return []
        
        related = []
        for neighbor, edge_data in self.graph[concept].items():
            if relation_type is None or edge_data.get('relation') == relation_type:
                related.append({
                    'concept': neighbor,
                    'weight': edge_data.get('weight', 1.0),
                    'relation': edge_data.get('relation', 'unknown')
                })
        
        # Sort by weight (strength of relation)
        related.sort(key=lambda x: x['weight'], reverse=True)
        
        return related[:limit]
    
    def get_concept_importance(self, limit=20):
        """Get the most important concepts in the knowledge graph"""
        if not self.concept_importance:
            return []
        
        # Sort concepts by importance
        sorted_concepts = sorted(self.concept_importance.items(), key=lambda x: x[1], reverse=True)
        
        return [{'concept': c[0], 'importance': c[1]} for c in sorted_concepts[:limit]]
    
    def get_concept_path(self, source, target, max_length=5):
        """Find the shortest path between two concepts"""
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return []
        
        try:
            path = nx.shortest_path(self.graph, source=source, target=target, weight='weight')
            
            # Limit path length
            if len(path) > max_length:
                return []
            
            # Get edge details
            path_with_details = []
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i+1]]
                path_with_details.append({
                    'source': path[i],
                    'target': path[i+1],
                    'weight': edge_data.get('weight', 1.0),
                    'relation': edge_data.get('relation', 'unknown')
                })
            
            return path_with_details
        except nx.NetworkXNoPath:
            return []
    
    def save_graph(self):
        """Save the knowledge graph to a file"""
        graph_file = os.path.join(self.graph_dir, 'knowledge_graph.pkl')
        with open(graph_file, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'concept_importance': self.concept_importance,
                'community_map': self.community_map
            }, f)
        logger.info(f"Saved knowledge graph to {graph_file}")
    
    def load_graph(self):
        """Load the knowledge graph from a file"""
        graph_file = os.path.join(self.graph_dir, 'knowledge_graph.pkl')
        if os.path.exists(graph_file):
            try:
                with open(graph_file, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.concept_importance = data['concept_importance']
                    self.community_map = data['community_map']
                logger.info(f"Loaded knowledge graph from {graph_file}")
                return True
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
        
        return False
    
    def visualize_graph(self, output_file=None, max_nodes=100):
        """Visualize the knowledge graph"""
        if not self.graph:
            logger.warning("Knowledge graph not built")
            return
        
        # Create a subgraph with limited nodes for visualization
        if len(self.graph) > max_nodes:
            # Get top concepts by importance
            top_concepts = sorted(self.concept_importance.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_concept_ids = [c[0] for c in top_concepts]
            subgraph = self.graph.subgraph(top_concept_ids)
        else:
            subgraph = self.graph
        
        # Set up the plot
        plt.figure(figsize=(12, 12))
        
        # Use spring layout for positioning
        pos = nx.spring_layout(subgraph, k=0.1, iterations=50)
        
        # Get node sizes based on importance
        node_sizes = [self.concept_importance.get(node, 0.1) * 5000 for node in subgraph.nodes()]
        
        # Get node colors based on community
        node_colors = [self.community_map.get(node, 0) for node in subgraph.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_size=node_sizes,
                              node_color=node_colors, 
                              alpha=0.8,
                              cmap=plt.cm.tab20)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, width=0.5, alpha=0.5)
        
        # Draw labels for top importance nodes
        top_n = min(20, len(subgraph))
        top_nodes = sorted([(n, self.concept_importance.get(n, 0)) 
                          for n in subgraph.nodes()], 
                          key=lambda x: x[1], 
                          reverse=True)[:top_n]
        labels = {node: node for node, _ in top_nodes}
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
        
        plt.title(f"AI Research Knowledge Graph (showing {len(subgraph)} concepts)")
        plt.axis('off')
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved knowledge graph visualization to {output_file}")
        else:
            output_file = os.path.join(self.graph_dir, 'knowledge_graph.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved knowledge graph visualization to {output_file}")
        
        plt.close()
    
    def get_adjacency_matrix(self, concepts=None):
        """Get adjacency matrix representation of the graph for model input"""
        if concepts is None:
            # Use all concepts
            concepts = list(self.graph.nodes())
        
        # Create mapping of concepts to indices
        concept_to_idx = {concept: i for i, concept in enumerate(concepts)}
        
        # Initialize adjacency matrix
        n = len(concepts)
        adj_matrix = np.zeros((n, n), dtype=np.float32)
        
        # Fill adjacency matrix
        for i, concept1 in enumerate(concepts):
            if concept1 not in self.graph:
                continue
                
            for concept2, edge_data in self.graph[concept1].items():
                if concept2 in concept_to_idx:
                    j = concept_to_idx[concept2]
                    adj_matrix[i, j] = edge_data.get('weight', 1.0)
        
        return torch.FloatTensor(adj_matrix), concept_to_idx
