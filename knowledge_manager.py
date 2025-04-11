from collections import Counter, defaultdict
import knowledge_graph # Assuming this module exists and has KnowledgeGraph class
from advanced_ai_analyzer import logger, CONFIG

class KnowledgeManager:
    """Manages knowledge representation: KG, clusters, retention, and gaps."""

    def __init__(self, engine):
        """Initialize with engine reference to access KB and state."""
        self.engine = engine # For accessing kb, iteration
        self.kb = engine.kb # Direct reference to knowledge base
        self.kg = knowledge_graph.KnowledgeGraph(self.kb) # Initialize KG

        # State Attributes moved from Engine
        self.concept_evolution = {} # Track how concepts evolve over time
        self.knowledge_gaps = set() # Track identified knowledge gaps
        self.concept_clusters = {} # Group related concepts
        self.research_frontiers = [] # Track cutting-edge research areas
        self.use_knowledge_graph = CONFIG.get('use_knowledge_graph', True)
        self.graph_built = False
        self.previous_concepts = set() # Track concepts from previous iteration
        self.knowledge_retention_score = 1.0 # Initialize retention score


    def update_knowledge_representations(self, query):
        """Update all managed knowledge representations after new papers are added."""
        self.update_concept_clusters()
        self.update_knowledge_graph() # Depends on clusters/concepts potentially
        self.apply_knowledge_retention()
        self.identify_knowledge_gaps(query)
        # Note: research_frontiers are updated within update_concept_clusters/update_knowledge_graph

    def update_knowledge_graph(self):
        """Update the knowledge graph with the latest papers."""
        if not self.use_knowledge_graph:
            return

        # Only rebuild graph periodically (e.g., every 10 papers or if not built)
        paper_count = len(self.kb.papers)
        # Check if paper_count modulo 10 is 0, or if graph is not built
        # Ensure paper_count is not 0 to avoid rebuilding initially
        if self.graph_built is False or (paper_count > 0 and paper_count % 10 == 0):
            logger.info("Building/updating knowledge graph")
            success = self.kg.build_graph()
            if success:
                self.graph_built = True
                try:
                     # Attempt visualization (might fail if dependencies missing)
                     self.kg.visualize_graph()
                except Exception as e:
                     logger.warning(f"Knowledge graph visualization failed: {e}")

                # Update research frontiers from graph communities
                communities = self.kg.get_concept_communities(top_n=5)
                if communities:
                    new_frontiers = []
                    for community in communities[:3]: # Use top 3 communities
                        if 'concepts' in community:
                             new_frontiers.extend(community['concepts'][:2])
                    if new_frontiers:
                        self.research_frontiers = list(set(new_frontiers)) # Keep unique
                        logger.info(f"Updated research frontiers from knowledge graph: {self.research_frontiers}")

    def update_concept_clusters(self):
        """Group related concepts into clusters based on co-occurrence."""
        logger.info("Updating concept clusters based on new papers")

        if len(self.kb.papers) < CONFIG.get('min_papers_for_clusters', 5):
            logger.info("Not enough papers to build meaningful concept clusters")
            return

        concept_counts = Counter()
        concept_co_occurrence = defaultdict(Counter)

        for paper_id, paper in self.kb.papers.items():
            paper_concepts = set(paper.get('concepts', []))
            if not paper_concepts:
                continue

            for concept in paper_concepts:
                concept_counts[concept] += 1

            for concept1 in paper_concepts:
                for concept2 in paper_concepts:
                    if concept1 != concept2:
                        concept_co_occurrence[concept1][concept2] += 1

        min_freq = CONFIG.get('min_concept_freq_for_clusters', 3)
        frequent_concepts = {concept for concept, count in concept_counts.items() if count >= min_freq}

        # Simple clustering: Group concepts strongly co-occurring with frequent concepts
        self.concept_clusters = {}
        visited = set()
        max_clusters = CONFIG.get('max_concept_clusters', 20)

        for concept in sorted(frequent_concepts, key=lambda c: concept_counts[c], reverse=True):
            if concept in visited:
                continue
            if len(self.concept_clusters) >= max_clusters:
                break

            cluster = {concept}
            visited.add(concept)

            # Find related concepts based on strong co-occurrence
            related = [(c, count) for c, count in concept_co_occurrence[concept].items()
                      if c in frequent_concepts and c not in visited and count >= CONFIG.get('min_cooccurrence_for_cluster', 2)]
            related.sort(key=lambda x: x[1], reverse=True)

            for related_concept, _ in related[:CONFIG.get('max_related_concepts_per_cluster', 5)]:
                cluster.add(related_concept)
                visited.add(related_concept)

            # Store cluster if it has more than one concept
            if len(cluster) > 1:
                 self.concept_clusters[concept] = list(cluster)

        logger.info(f"Created {len(self.concept_clusters)} concept clusters")

        # Update research frontiers based on cluster keys (most frequent concepts in clusters)
        self.research_frontiers = list(self.concept_clusters.keys())[:CONFIG.get('num_frontiers_from_clusters', 5)]

    def apply_knowledge_retention(self):
        """Apply knowledge retention mechanisms to prevent catastrophic forgetting."""
        logger.info("Applying knowledge retention mechanisms")

        if len(self.kb.papers) < 10:
            return # Not enough history

        # Get current concepts present in the KB
        current_concepts = set()
        for paper_id, paper in self.kb.papers.items():
            current_concepts.update(paper.get('concepts', []))

        retention_score = 1.0 # Default if no previous concepts
        if self.previous_concepts:
            overlap = len(current_concepts.intersection(self.previous_concepts))
            union = len(current_concepts.union(self.previous_concepts))
            if union > 0:
                retention_score = overlap / union
            else: # Handle case where both sets might be empty
                 retention_score = 1.0

        # Store current concepts for the next iteration
        self.previous_concepts = current_concepts

        # Update smoothed retention score
        smoothing_factor = CONFIG.get('retention_score_smoothing', 0.7)
        self.knowledge_retention_score = (smoothing_factor * self.knowledge_retention_score) + ((1 - smoothing_factor) * retention_score)
        logger.info(f"Knowledge retention score: {self.knowledge_retention_score:.4f}")
        # Update engine's score if needed elsewhere (optional)
        # self.engine.knowledge_retention_score = self.knowledge_retention_score

    def identify_knowledge_gaps(self, query):
        """Identify knowledge gaps based on processed papers and query."""
        try:
            # Simple approach: concepts in query not well represented in KB
            query_tokens = set(query.lower().split()) # Basic tokenization
            # Consider using PaperProcessor.preprocess for better concept extraction from query

            covered_concepts_tokens = set()
            for concept in self.kb.concept_index.keys():
                 covered_concepts_tokens.update(concept.lower().split())

            # Identify tokens in query but not in KB concepts
            gaps = query_tokens - covered_concepts_tokens
            # Filter out common stop words if necessary
            # from nltk.corpus import stopwords
            # stop_words = set(stopwords.words('english'))
            # gaps = {gap for gap in gaps if gap not in stop_words and len(gap) > 2}

            if gaps:
                logger.info(f"Identified potential knowledge gap terms: {gaps}")
                self.knowledge_gaps.update(gaps) # Add new gaps

            # Limit the size of stored knowledge gaps
            max_gaps = CONFIG.get('max_knowledge_gaps', 50)
            if len(self.knowledge_gaps) > max_gaps:
                 # Keep the most recently identified ones (simple approach)
                 self.knowledge_gaps = set(list(self.knowledge_gaps)[-max_gaps:])

            return gaps
        except Exception as e:
             logger.error(f"Error identifying knowledge gaps: {e}")
             return set()

    def identify_research_frontiers(self):
        """Identify emerging research frontiers based on knowledge graph and concept clusters.
        
        Returns:
            list: Dictionary objects containing frontier concepts and their metadata
        """
        logger.info("Identifying research frontiers")
        
        frontiers = []
        
        # Method 1: Use existing frontiers from concept clusters or graph
        if self.research_frontiers:
            logger.info(f"Using {len(self.research_frontiers)} existing research frontiers")
            
        # Method 2: Analyze concept growth rates
        concept_growth = {}
        if len(self.kb.papers) >= 10:  # Need enough papers for meaningful analysis
            # Count concepts by time period (e.g., by published date)
            concept_by_time = defaultdict(Counter)
            
            # Group papers by time period (using simplified approach)
            papers_by_time = defaultdict(list)
            for paper_id, paper in self.kb.papers.items():
                # Use year-month as period or iteration if no date
                if 'metadata' in paper and 'published' in paper['metadata']:
                    time_period = paper['metadata']['published'][:7]  # YYYY-MM format
                else:
                    time_period = f"iter-{self.engine.iteration}"
                    
                papers_by_time[time_period].append(paper)
            
            # Sort time periods
            sorted_periods = sorted(papers_by_time.keys())
            
            # Count concepts in each period
            for period in sorted_periods:
                for paper in papers_by_time[period]:
                    concepts = paper.get('concepts', [])
                    concept_by_time[period].update(concepts)
            
            # Calculate growth rate if we have multiple periods
            if len(sorted_periods) >= 2:
                recent_period = sorted_periods[-1]
                prev_period = sorted_periods[-2]
                
                # Find concepts with significant growth
                for concept, recent_count in concept_by_time[recent_period].items():
                    prev_count = concept_by_time[prev_period].get(concept, 0)
                    if prev_count > 0:
                        growth = (recent_count - prev_count) / prev_count
                    else:
                        growth = float(recent_count)  # New concept
                        
                    concept_growth[concept] = growth
        
        # Method 3: Use citation density from knowledge graph if available
        citation_importance = {}
        if self.use_knowledge_graph and self.graph_built:
            try:
                # Get centrality metrics from knowledge graph
                centrality = self.kg.get_concept_centrality(top_n=20)
                citation_importance = {item['concept']: item['score'] for item in centrality}
            except Exception as e:
                logger.error(f"Error getting centrality from knowledge graph: {e}")
        
        # Combine methods to get final frontiers
        # 1. Start with existing frontiers
        frontier_concepts = set(self.research_frontiers)
        
        # 2. Add top growth concepts
        growth_threshold = CONFIG.get('frontier_growth_threshold', 0.5)
        growing_concepts = {c for c, g in concept_growth.items() if g > growth_threshold}
        frontier_concepts.update(growing_concepts)
        
        # 3. Add important concepts from citation analysis
        citation_threshold = CONFIG.get('frontier_citation_threshold', 0.4)
        important_cited = {c for c, s in citation_importance.items() if s > citation_threshold}
        frontier_concepts.update(important_cited)
        
        # Create structured frontier objects
        for concept in frontier_concepts:
            frontier = {
                'concept': concept,
                'growth_rate': concept_growth.get(concept, 0.0),
                'citation_importance': citation_importance.get(concept, 0.0),
                'cluster': next((cluster for leader, cluster in self.concept_clusters.items() 
                                 if concept in cluster), []),
                'papers': [pid for pid, p in self.kb.papers.items() 
                          if concept in p.get('concepts', [])][:5]  # Limit to 5 papers
            }
            frontiers.append(frontier)
        
        # Sort by combined importance
        for f in frontiers:
            f['importance'] = (f['growth_rate'] * 0.6) + (f['citation_importance'] * 0.4)
            
        frontiers.sort(key=lambda x: x['importance'], reverse=True)
        
        # Keep only top N frontiers
        max_frontiers = CONFIG.get('max_research_frontiers', 10)
        frontiers = frontiers[:max_frontiers]
        
        # Update the class attribute
        self.research_frontiers = [f['concept'] for f in frontiers]
        
        logger.info(f"Identified {len(frontiers)} research frontiers")
        return frontiers
