from advanced_ai_analyzer import *
from advanced_ai_analyzer_paper_processor import PaperProcessor
from advanced_ai_analyzer_knowledge_base import KnowledgeBase
from advanced_ai_analyzer_learning import LearningSystem
from collections import Counter, defaultdict
import transfer_learning
import knowledge_graph
import transformer_models
import torch
import itertools
from utils.gpu_manager import GPUMonitor

class RecursiveResearchEngine:
    """Advanced recursive engine for self-iterating AI research"""
    
    def __init__(self, papers_dir=CONFIG['papers_dir'], models_dir=CONFIG['models_dir']):
        self.processor = PaperProcessor(papers_dir)
        self.kb = KnowledgeBase()
        self.learner = LearningSystem(self.kb, models_dir)
        self.iteration = 0
        self.performance_history = []
        self.research_focus = []
        self.exploration_rate = 0.3  # Initial exploration rate
        self.model_types = ['enhanced', 'simple']
        self.current_model = 'enhanced'
        self.meta_learning_data = []
        
        # Advanced tracking metrics
        self.concept_evolution = {}  # Track how concepts evolve over time
        self.research_trajectory = []  # Track the research path
        self.knowledge_gaps = set()  # Track identified knowledge gaps
        self.concept_clusters = {}  # Group related concepts
        self.research_frontiers = []  # Track cutting-edge research areas
        
        # Initialize knowledge graph
        self.kg = knowledge_graph.KnowledgeGraph(self.kb)
        self.use_knowledge_graph = CONFIG.get('use_knowledge_graph', True)
        self.graph_built = False
        
        # Self-improvement parameters
        self.learning_rate_schedule = {}
        self.model_performance_by_domain = {}
        self.adaptive_exploration = True
        self.knowledge_retention_score = 1.0
        
        # Create directories for research artifacts
        self.reports_dir = os.path.join(models_dir, 'reports')
        self.concept_maps_dir = os.path.join(models_dir, 'concept_maps')
        self.research_frontiers_dir = os.path.join(models_dir, 'frontiers')
        
        for directory in [self.reports_dir, self.concept_maps_dir, self.research_frontiers_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def process_papers(self, query, max_results=1000):
        batch_size = 32  # Initial batch size
        
        # Use the new method to download papers from the Git repository
        # The 'query' argument is ignored when fetching from the repo.
        papers = self.processor.download_papers_from_repo(max_results=max_results)
        
        if not papers: # Handle case where repo fetching fails or returns no papers
            logger.warning("No papers retrieved from the repository. Skipping processing.")
            return 0
            
        paper_stream = iter(papers)   # create an iterator from the downloaded papers
        processed_count = 0
        
        while True:
            # Dynamically adjust batch size
            batch_size = GPUMonitor.optimize_batch_size(batch_size)
            
            batch = list(itertools.islice(paper_stream, batch_size))
            if not batch:
                break
                
            processed = self.processor.process_papers_batch(batch)
            self.kb.add_papers(processed)
            torch.cuda.empty_cache()
            
            processed_count += len(processed)
            
            # Dynamically adjust batch size based on memory
            if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                batch_size = max(4, batch_size // 2)  # Reduce batch size but keep minimum
            
        return processed_count  # Return the count of processed papers (an integer)
    
    def _select_papers(self, query_results):
        """Select most relevant papers using:
        1. Citation count
        2. Recency
        3. Concept novelty score
        """
        scored_papers = []
        for paper in query_results:
            # Calculate composite score (0-1 range)
            recency = 1 - (datetime.now() - paper['published']).days / 365  # 1 year normalization
            citations = min(paper.get('citation_count', 0) / 100, 1)  # Normalized to 100 citations
            novelty = self.kb.concept_novelty_score(paper['concepts'])
            
            score = 0.4*recency + 0.4*citations + 0.2*novelty
            scored_papers.append((score, paper))
            
        # Return top 20% highest scoring papers
        scored_papers.sort(reverse=True)
        return [p[1] for p in scored_papers[:int(len(scored_papers)*0.2)]]
    
    def _generate_research_query(self):
        """Generate a more specific, field-focused research query"""
        # If first iteration or no performance data, use initial query
        if self.iteration == 0 or not self.performance_history:
            if self.research_focus and isinstance(self.research_focus[0], dict):
                # Use structured field-specific query if available
                focus = self.research_focus[0]
                field = focus.get('field', 'artificial intelligence')
                subfield = focus.get('subfield', '')
                topics = focus.get('topics', [])
                
                # Construct specific query
                query_parts = [field]
                if subfield:
                    query_parts.append(subfield)
                if topics:
                    query_parts.extend(topics[:3])  # Limit to 3 topics for focus
                
                return " ".join(query_parts)
            else:
                return self.research_focus[0] if self.research_focus else "artificial intelligence"
        
        # Get latest evaluation report
        latest_eval = self.performance_history[-1]
        
        # Find worst performing category - this often corresponds to a research field
        if 'per_category' in latest_eval:
            categories = []
            f1_scores = []
            
            for category, metrics in latest_eval['per_category'].items():
                # Only consider categories with enough samples
                if metrics['support'] >= 5:
                    categories.append(category)
                    f1_scores.append(metrics['f1'])
            
            if categories:
                # Sometimes explore random categories for diversity
                if random.random() < self.exploration_rate:
                    target_category = random.choice(categories)
                    logger.info(f"Exploring random category: {target_category}")
                else:
                    # Find worst performing category
                    worst_idx = np.argmin(f1_scores)
                    target_category = categories[worst_idx]
                    logger.info(f"Focusing on worst performing category: {target_category}")
                
                # Get top concepts for this category
                concepts = self.kb.get_top_concepts(target_category, n=5)
                
                # Add relevant field-specific terms based on the category
                # Map common arXiv categories to specific research fields
                field_terms = {
                    'cs.AI': ['artificial intelligence', 'reasoning', 'knowledge representation'],
                    'cs.LG': ['machine learning', 'neural networks', 'deep learning'],
                    'cs.CL': ['natural language processing', 'language models', 'text generation'],
                    'cs.CV': ['computer vision', 'image recognition', 'object detection'],
                    'cs.RO': ['robotics', 'reinforcement learning', 'control systems'],
                    'stat.ML': ['statistical learning', 'probabilistic models', 'bayesian methods']
                }
                
                # Add field-specific terms to the query
                field_specific_terms = []
                for field, terms in field_terms.items():
                    if field.lower() in target_category.lower() or any(term.lower() in target_category.lower() for term in terms):
                        field_specific_terms = terms
                        break
                
                # Generate query combining category, field terms, and concepts
                query_parts = [target_category]
                if field_specific_terms:
                    query_parts.append(random.choice(field_specific_terms))
                query_parts.extend(concepts[:3])  # Limit to top 3 concepts for focus
                
                return " ".join(query_parts)
        
        # Fallback to using top overall concepts with a field focus
        concepts = self.kb.get_top_concepts(n=5)
        
        # Try to identify a field from the concepts
        ai_fields = ["machine learning", "deep learning", "reinforcement learning", 
                     "natural language processing", "computer vision", "robotics",
                     "knowledge representation", "planning", "reasoning", "ethics",
                     "neural networks", "generative models", "transformer models",
                     "multimodal learning", "autonomous systems"]
        
        detected_fields = []
        for field in ai_fields:
            for concept in concepts:
                if field.lower() in concept.lower():
                    detected_fields.append(field)
                    break
        
        # Construct a field-focused query
        query_parts = []
        if detected_fields:
            query_parts.append(detected_fields[0])  # Use the first detected field
        query_parts.extend(concepts[:3])  # Add top concepts, limiting to 3
        
        return " ".join(query_parts)
    
    def _update_citation_network(self, metadata):
        """Update the citation network with new paper metadata
        
        Args:
            metadata: List of paper metadata dictionaries
        """
        if not hasattr(self.kb, 'citation_network'):
            self.kb.citation_network = {}
            
        # Initialize dictionaries if not present
        if not hasattr(self.kb, 'author_papers'):
            self.kb.author_papers = defaultdict(set)
            
        if not hasattr(self.kb, 'category_papers'):
            self.kb.category_papers = defaultdict(set)
            
        # Process each paper to update citation network
        for paper in metadata:
            paper_id = paper.get('id')
            if not paper_id:
                continue
                
            # Initialize this paper in the citation network if not present
            if paper_id not in self.kb.citation_network:
                self.kb.citation_network[paper_id] = {
                    'citations': set(),   # Papers this paper cites
                    'cited_by': set(),    # Papers that cite this paper
                    'authors': set(),     # Authors of this paper
                    'categories': set()   # Categories of this paper
                }
            
            # Add authors
            authors = paper.get('authors', [])
            for author in authors:
                author_name = author.get('name', '')
                if author_name:
                    # Add author to paper's author list
                    self.kb.citation_network[paper_id]['authors'].add(author_name)
                    # Add paper to author's paper list
                    self.kb.author_papers[author_name].add(paper_id)
            
            # Add categories
            categories = paper.get('categories', [])
            for category in categories:
                # Add category to paper's category list
                self.kb.citation_network[paper_id]['categories'].add(category)
                # Add paper to category's paper list
                self.kb.category_papers[category].add(paper_id)
            
            # Process citations if available
            citations = paper.get('citations', [])
            for citation in citations:
                # Add citation to this paper's outgoing citations
                self.kb.citation_network[paper_id]['citations'].add(citation)
                
                # Add this paper to the cited paper's incoming citations
                if citation not in self.kb.citation_network:
                    self.kb.citation_network[citation] = {
                        'citations': set(),
                        'cited_by': set(),
                        'authors': set(),
                        'categories': set()
                    }
                self.kb.citation_network[citation]['cited_by'].add(paper_id)
    
    def _apply_transfer_learning(self, query):
        """Apply transfer learning when no new papers are found
        
        This leverages knowledge from related domains to improve the model
        even when no direct papers are available.
        
        Args:
            query: The query for which no new papers were found
        """
        # Call the existing transfer learning implementation
        transfer_learning.apply_transfer_learning(self, query)
    
    def _calculate_adaptive_batch_size(self):
        """Calculate adaptive batch size based on current knowledge and iteration"""
        base_size = CONFIG.get('papers_per_batch', 10)
        # Increase batch size as we progress through iterations
        scaling_factor = min(2.0, 1.0 + (self.iteration / 10.0))
        # Adjust based on knowledge retention
        knowledge_factor = max(1.0, self.knowledge_retention_score * 1.5)
        
        adaptive_size = int(base_size * scaling_factor * knowledge_factor)
        logger.info(f"Using adaptive batch size: {adaptive_size} papers")
        return adaptive_size
        
    def _select_best_model(self):
        """Select the best performing model type based on evaluation"""
        if len(self.performance_history) < len(self.model_types):
            # Not enough data to compare all models
            return
        
        # Get latest performance for each model type
        model_performances = {}
        for eval_data in reversed(self.performance_history):
            if eval_data['model_type'] not in model_performances:
                model_performances[eval_data['model_type']] = eval_data['overall_f1']
            
            if len(model_performances) == len(self.model_types):
                break
        
        # Select best model
        best_model = max(model_performances.items(), key=lambda x: x[1])[0]
        if best_model != self.current_model:
            self.current_model = best_model
            logger.info(f"Switching to better performing model: {self.current_model}")
    
    def _identify_knowledge_gaps(self, query):
        """Identify knowledge gaps based on processed papers and query"""
        # Extract key concepts from the query
        query_concepts = set(query.lower().split())
        
        # Check coverage of these concepts in our knowledge base
        covered_concepts = set()
        for concept in self.kb.concept_index.keys():
            covered_concepts.update(concept.lower().split())
        
        # Identify gaps
        gaps = query_concepts - covered_concepts
        if gaps:
            logger.info(f"Identified knowledge gaps: {gaps}")
            self.knowledge_gaps.update(gaps)
        
        return gaps
        
    def _generate_meta_learning_data(self, query, processed_count, eval_results):
        """Generate meta-learning data from research iteration"""
        self.meta_learning_data.append({
            'iteration': self.iteration,
            'query': query,
            'papers_processed': processed_count,
            'model_type': self.current_model,
            'accuracy': eval_results['accuracy'],
            'f1_score': eval_results['f1_score'],
            'exploration_rate': self.exploration_rate,
            'timestamp': time.time()
        })
    
    def _generate_research_report(self):
        """Generate a comprehensive research report based on all papers"""
        # Create research report with findings
        report = "# AI Research Report\n\n"
        
        # Add executive summary
        report += "## Executive Summary\n\n"
        report += f"This report summarizes the findings from analyzing {len(self.kb.papers)} research papers "
        report += "in the field of artificial intelligence and machine learning.\n\n"
        
        # Add key metrics
        report += "### Key Metrics\n\n"
        report += f"* Papers analyzed: {len(self.kb.papers)}\n"
        report += f"* Unique concepts identified: {self.kb.count_concepts()}\n"
        report += f"* Research frontiers: {len(self.research_frontiers)}\n"
        report += f"* Training iterations: {self.iteration}\n\n"
        
        # Add section on research frontiers
        report += "## Research Frontiers\n\n"
        report += "The following research areas represent the cutting edge of AI research:\n\n"
        
        if self.research_frontiers:
            for i, frontier in enumerate(self.research_frontiers[:5]):
                # Get related concepts if we have a knowledge graph
                related = []
                if self.graph_built:
                    related_concepts = self.kg.get_related_concepts(frontier, limit=3)
                    related = [rc['concept'] for rc in related_concepts]
                
                report += f"### {i+1}. {frontier.title()}\n"
                if related:
                    report += f"Related concepts: {', '.join(related)}\n\n"
                else:
                    report += "\n"
                
                # Add papers in this frontier
                papers_in_frontier = []
                for paper_id, paper in self.kb.papers.items():
                    if 'concepts' in paper and frontier in paper['concepts']:
                        papers_in_frontier.append(paper)
                
                if papers_in_frontier:
                    # Sort by recency or importance
                    if 'published' in papers_in_frontier[0]:
                        papers_in_frontier.sort(key=lambda p: p.get('published', ''), reverse=True)
                    
                    # List top papers
                    report += "Key papers:\n"
                    for paper in papers_in_frontier[:3]:
                        report += f"* {paper.get('title', 'Unknown')}\n"
                    report += "\n"
        else:
            report += "No clear research frontiers identified yet. More papers needed.\n\n"
        
        # Add knowledge graph insights if available
        if self.graph_built:
            report += "## Knowledge Graph Insights\n\n"
            
            # Add community detection results
            communities = self.kg.get_concept_communities(top_n=3)
            if communities:
                report += "### Concept Communities\n\n"
                report += "The research has identified the following concept communities:\n\n"
                
                for i, community in enumerate(communities[:3]):
                    report += f"**Community {i+1}**: "
                    report += f"{', '.join(community['concepts'][:5])}\n\n"
            
            # Add central concepts (highest betweenness centrality)
            key_concepts = self.kg.get_concept_importance(limit=10)
            if key_concepts:
                report += "### Central Concepts\n\n"
                report += "These concepts are central to the research field:\n\n"
                for concept in key_concepts[:5]:
                    report += f"* {concept['concept']} (importance: {concept['importance']:.3f})\n"
                report += "\n"
        
        # Add model performance section
        report += "## Model Performance\n\n"
        
        if self.performance_history:
            report += "### Learning Progress\n\n"
            report += "Performance metrics over time:\n\n"
            
            report += "| Iteration | F1 Score | Accuracy |\n"
            report += "|-----------|----------|----------|\n"
            
            for i, perf in enumerate(self.performance_history[-5:]):  # Last 5 iterations
                iter_num = len(self.performance_history) - 5 + i + 1
                f1 = perf.get('overall_f1', 0)
                acc = perf.get('accuracy', 0)
                report += f"| {iter_num} | {f1:.3f} | {acc:.3f} |\n"
            
            report += "\n"
            
            # Add best performing model
            report += f"Best performing model: {self.current_model}\n\n"
        else:
            latest_eval = None
            performance_trend = []
        
        # Generate knowledge base summary
        kb_summary = {
            'total_papers': len(self.kb.papers),
            'concepts': len(self.kb.get_top_concepts(n=None)),
            'categories': len(self.kb.category_index),
            'authors': len(self.kb.author_index),
            'latest_paper': max(p['published'] for p in self.kb.papers.values() if 'published' in p) if any('published' in p for p in self.kb.papers.values()) else None
        }
        
        # Create report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'iteration': self.iteration,
            'total_papers': len(self.kb.papers),
            'knowledge_summary': kb_summary,
            'performance': latest_eval,
            'performance_trend': performance_trend,
            'meta_learning_data': self.meta_learning_data[-5:] if self.meta_learning_data else [],
            'current_model': self.current_model,
            'exploration_rate': self.exploration_rate
        }
        
        # Save report
        report_path = os.path.join(self.reports_dir, f'research_report_iter_{self.iteration}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated research report: {report_path}")
        return report
    
    def run_iteration(self, query):
        """Run single research iteration with error handling"""
        self.iteration += 1
        logger.info(f"\n==================================================\nStarting Iteration {self.iteration}\n==================================================")
        logger.info(f"Research query: '{query}'")
        
        # Track research trajectory
        self.research_trajectory.append({
            'iteration': self.iteration,
            'query': query,
            'timestamp': time.time()
        })
        
        # Process papers for query with adaptive batch size
        batch_size = self._calculate_adaptive_batch_size()
        papers_added = self.process_papers(query, max_results=batch_size)
        
        # Train and evaluate - if no new papers, apply transfer learning
        if papers_added > 0:
            logger.info(f"Added {papers_added} new papers to knowledge base")
            
            # Update concept clusters based on new papers
            self._update_concept_clusters()
            
            # Apply curriculum learning strategy
            self._apply_curriculum_learning()
            
            # Track progress toward target paper count
            total_papers = len(self.kb.papers)
            target_papers = CONFIG.get('max_papers_total', 1000)
            progress_pct = (total_papers / target_papers) * 100
            logger.info(f"Progress: {total_papers}/{target_papers} papers collected ({progress_pct:.1f}%)")
            
            # Train the model with adaptive learning parameters
            logger.info(f"Training {self.current_model} model with adaptive parameters...")
            learning_rate = self._calculate_adaptive_learning_rate()
            self.learner.train(model_type=self.current_model, learning_rate=learning_rate)
            
            # Evaluate model performance with comprehensive metrics
            logger.info(f"Evaluating {self.current_model} model with comprehensive metrics")
            eval_results = self.learner.evaluate(model_type=self.current_model)
            
            if eval_results is None:
                logger.warning("Evaluation failed - using default metrics")
                eval_results = {
                    'overall_f1': 0.5,
                    'accuracy': 0.5,
                    'status': 'evaluation_failed'
                }
            
            # Record performance in history
            self.performance_history.append({
                'iteration': self.iteration,
                'papers_added': papers_added,
                'query': query,
                'model_type': self.current_model,
                'metrics': eval_results
            })
            
            logger.info(f"Completed iteration {self.iteration} with {papers_added} new papers")
            if 'overall_f1' in eval_results:
                logger.info(f"Model F1 score: {eval_results['overall_f1']:.4f}")
                
            return eval_results
        else:
            logger.warning(f"No new papers added for query: '{query}', attempting transfer learning")
            transfer_results = self._apply_transfer_learning(query)
            
            if transfer_results['success']:
                logger.info(f"Applied transfer learning successfully")
                eval_results = transfer_results['performance']
                
                # Record transfer learning performance
                self.performance_history.append({
                    'iteration': self.iteration,
                    'papers_added': 0,
                    'papers_transferred': transfer_results['papers_transferred'],
                    'query': query,
                    'model_type': self.current_model,
                    'transfer_learning': True,
                    'metrics': eval_results
                })
                
                logger.info(f"Completed iteration {self.iteration} with transfer learning")
                if 'overall_f1' in eval_results:
                    logger.info(f"Model F1 score after transfer: {eval_results['overall_f1']:.4f}")
                
                return eval_results
            else:
                logger.warning("Transfer learning failed to improve the model")
                return {"error": "No new papers added and transfer learning failed"}
        
        # Track domain-specific performance
        self._track_domain_performance(query, eval_results)
        
        # Generate meta-learning data and update research frontiers
        if eval_results:
            self._generate_meta_learning_data(query, papers_added, eval_results)
            self._update_research_frontiers(query, eval_results)
        
        # Generate learning curves and visualizations
        try:
            self.learner.generate_learning_curves()
            self._generate_concept_map()
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")
        
        # Generate comprehensive research report with increasing frequency as system improves
        report_frequency = max(1, int(5 * (1 - self.knowledge_retention_score)))
        if self.iteration % report_frequency == 0:
            self._generate_research_report()
        
        # Update exploration rate with decay
        self._update_exploration_rate(eval_results)
        
        # Apply knowledge retention mechanisms
        self._apply_knowledge_retention()
        
        # Generate next research query with adaptive exploration
        next_query = self._generate_next_query(query, use_adaptive=self.adaptive_exploration)
        
        # Periodically attempt model architecture improvements
        if self.iteration % 5 == 0:
            self._attempt_model_improvement()
            
        return next_query
    
    def _enhance_query_for_ai_categories(self, query):
        """Enhance query to ONLY target core AI/ML categories"""
        from config import CONFIG
        return f"({query}) AND ({' OR '.join(f'cat:{cat}' for cat in CONFIG['ai_categories'])})"
    
    def _update_knowledge_graph(self):
        """Update the knowledge graph with the latest papers
        
        This builds or updates the graph representation of concepts and their relationships
        """
        if not self.use_knowledge_graph:
            return
            
        # Only rebuild graph periodically (every 10 new papers or if not built yet)
        paper_count = len(self.kb.papers)
        if not self.graph_built or paper_count % 10 == 0:
            logger.info("Building/updating knowledge graph")
            success = self.kg.build_graph()
            if success:
                self.graph_built = True
                
                # Generate visualization
                self.kg.visualize_graph()
                
                # Update research frontiers from graph communities
                communities = self.kg.get_concept_communities(top_n=5)
                if communities:
                    # Extract key concepts from top communities
                    self.research_frontiers = []
                    for community in communities[:3]:  # Use top 3 communities
                        self.research_frontiers.extend(community['concepts'][:2])  # Take top 2 concepts from each
                    
                    logger.info(f"Updated research frontiers from knowledge graph: {self.research_frontiers}")
    
    def _update_concept_clusters(self):
        """Group related concepts into clusters based on co-occurrence and semantic similarity
        
        This helps identify conceptual themes across the research literature
        """
        logger.info("Updating concept clusters based on new papers")
        
        # Skip if not enough papers yet
        if len(self.kb.papers) < 5:
            logger.info("Not enough papers to build meaningful concept clusters")
            return
        
        # Get concepts and their co-occurrence patterns
        concept_counts = Counter()
        concept_co_occurrence = defaultdict(Counter)
        
        # Count concepts and their co-occurrences across papers
        for paper_id, paper in self.kb.papers.items():
            if 'concepts' not in paper or not paper['concepts']:
                continue
                
            # Get unique concepts in this paper
            paper_concepts = set(paper['concepts'])
            
            # Update overall concept counts
            for concept in paper_concepts:
                concept_counts[concept] += 1
            
            # Update co-occurrence counts
            for concept1 in paper_concepts:
                for concept2 in paper_concepts:
                    if concept1 != concept2:
                        concept_co_occurrence[concept1][concept2] += 1
        
        # Only consider frequent concepts (mentioned in at least 3 papers)
        min_freq = 3
        frequent_concepts = {concept for concept, count in concept_counts.items() if count >= min_freq}
        
        # Group concepts into clusters
        self.concept_clusters = {}
        visited = set()
        
        # Start with the most frequent concepts
        for concept in sorted(frequent_concepts, key=lambda c: concept_counts[c], reverse=True):
            if concept in visited:
                continue
                
            # Start a new cluster
            cluster = [concept]
            visited.add(concept)
            
            # Find related concepts based on co-occurrence
            related = [(c, count) for c, count in concept_co_occurrence[concept].items() 
                      if c in frequent_concepts and c not in visited]
            related.sort(key=lambda x: x[1], reverse=True)
            
            # Add top related concepts to this cluster
            for related_concept, _ in related[:5]:  # Limit to top 5 related concepts
                cluster.append(related_concept)
                visited.add(related_concept)
            
            # Store cluster with a representative name
            self.concept_clusters[concept] = cluster
            
            # Limit the number of clusters for efficiency
            if len(self.concept_clusters) >= 20:
                break
                
        logger.info(f"Created {len(self.concept_clusters)} concept clusters")
        
        # Store the top research themes
        self.research_frontiers = list(self.concept_clusters.keys())[:5]
        
    def _apply_knowledge_retention(self):
        """Apply knowledge retention mechanisms to prevent catastrophic forgetting
        
        This ensures the system retains important concepts even as it learns new ones
        """
        logger.info("Applying knowledge retention mechanisms")
        
        # Skip if not enough papers yet
        if len(self.kb.papers) < 10:
            return
            
        # Calculate retention score based on concept stability
        retention_score = 1.0
        
        # Track concept frequency across iterations to measure stability
        current_concepts = set()
        for paper_id, paper in self.kb.papers.items():
            if 'concepts' in paper and paper['concepts']:
                current_concepts.update(paper['concepts'])
        
        # Calculate retention based on concept overlap with previous iterations
        if hasattr(self, 'previous_concepts') and self.previous_concepts:
            # Jaccard similarity between current and previous concept sets
            overlap = len(current_concepts.intersection(self.previous_concepts))
            union = len(current_concepts.union(self.previous_concepts))
            if union > 0:
                retention_score = overlap / union
        
        # Store current concepts for next iteration
        self.previous_concepts = current_concepts
        
        # Update retention score (with smoothing to prevent wild fluctuations)
        self.knowledge_retention_score = 0.7 * self.knowledge_retention_score + 0.3 * retention_score
        logger.info(f"Knowledge retention score: {self.knowledge_retention_score:.4f}")
        
    def _apply_curriculum_learning(self):
        """Apply curriculum learning to gradually increase task complexity
        
        This helps the model learn more efficiently by starting with simpler examples
        and gradually increasing complexity.
        """
        # Skip if no model yet
        if not hasattr(self.learner, 'current_model') or self.learner.current_model is None:
            return
            
        # Determine current curriculum stage based on iteration number
        stage = min(3, self.iteration // 5)  # Stages 0, 1, 2, 3 at iterations 0, 5, 10, 15+
        
        if stage == 0:
            # Early stage: Focus on papers with clear categories and simpler concepts
            logger.info("Curriculum Stage 0: Focusing on core concepts and clear categories")
            # Use default settings for early stages
            return
            
        elif stage == 1:
            # Middle stage: Incorporate more diverse papers and categories
            logger.info("Curriculum Stage 1: Incorporating diverse papers and concepts")
            # Adjust learning parameters for broader learning
            self.exploration_rate = 0.4  # Increase exploration
            return
            
        elif stage == 2:
            # Advanced stage: Focus on emerging research themes and interdisciplinary connections
            logger.info("Curriculum Stage 2: Focusing on research themes and connections")
            # Adjust for interdisciplinary learning
            self.exploration_rate = 0.3  # Balance exploration and exploitation
            return
            
        else:  # stage == 3
            # Expert stage: Focus on knowledge gaps and cutting-edge research
            logger.info("Curriculum Stage 3: Targeting knowledge gaps and cutting-edge research")
            # Full advanced learning mode
            self.exploration_rate = 0.2  # More exploitation of knowledge
            return
            
    def build_knowledge_graph(self):
        """Build a knowledge graph from the current knowledge base
        
        Returns:
            Boolean indicating success
        """
        if not self.use_knowledge_graph:
            return False
            
        success = self.kg.build_graph()
        if success:
            self.graph_built = True
            self.kg.visualize_graph()
            
            # Update research frontiers
            communities = self.kg.get_concept_communities(top_n=5)
            if communities:
                # Extract top concepts from each community
                self.research_frontiers = []
                for community in communities[:3]:  # Use top 3 communities
                    self.research_frontiers.extend(community['concepts'][:2])  # Take top 2 concepts from each
                    
            return True
        return False
    
    def get_research_map(self):
        """Generate a research map based on current knowledge
        
        Returns:
            Dictionary with research map data
        """
        research_map = {
            'frontiers': [],
            'clusters': [],
            'key_concepts': [],
            'citation_hubs': []
        }
        
        # Add research frontiers
        if self.research_frontiers:
            research_map['frontiers'] = self.research_frontiers
            
        # Add concept clusters
        if self.concept_clusters:
            for key, cluster in list(self.concept_clusters.items())[:5]:  # Top 5 clusters
                research_map['clusters'].append({
                    'name': key,
                    'concepts': cluster
                })
                
        # Add key concepts from knowledge graph
        if self.graph_built:
            research_map['key_concepts'] = [item['concept'] for item in self.kg.get_concept_importance(limit=20)]
            
        # Add citation hubs (papers that are highly cited)
        if hasattr(self.kb, 'citation_network') and self.kb.citation_network:
            citation_counts = Counter()
            for paper_id, paper in self.kb.papers.items():
                if 'citations' in paper:
                    for cited in paper['citations']:
                        citation_counts[cited] += 1
                        
            # Get top cited papers
            for paper_id, count in citation_counts.most_common(10):
                if paper_id in self.kb.papers:
                    research_map['citation_hubs'].append({
                        'id': paper_id,
                        'title': self.kb.papers[paper_id].get('title', 'Unknown'),
                        'citations': count
                    })
                    
        return research_map
    
    def run(self, initial_queries, iterations=5, max_results=1000):
        """Run the research engine with field-specific queries
        
        Args:
            initial_queries: List of queries to use. Can be either:
                - Strings for simple queries (e.g., "machine learning")
                - Dictionaries for structured field-specific queries with keys:
                  'field' (required): Main research field (e.g., "machine learning")
                  'subfield' (optional): Specific subfield (e.g., "deep reinforcement learning")
                  'topics' (optional): List of specific topics (e.g., ["transformers", "attention mechanisms"])
            iterations: Number of iterations to run
            max_results: Maximum number of papers to process per query
            
        Returns:
            Summary of research findings
        """
        # Store the research focus for use in _generate_research_query
        self.research_focus = initial_queries
        
        # Track total papers processed
        total_papers_processed = 0
        
        # Main research loop
        for i in range(iterations):
            logger.info(f"Starting research iteration {i+1}/{iterations}")
            
            # Process each query
            for query_spec in initial_queries:
                # Convert query specification to string query
                if isinstance(query_spec, dict):
                    # Handle structured field-specific query
                    field = query_spec.get('field', '')
                    subfield = query_spec.get('subfield', '')
                    topics = query_spec.get('topics', [])
                    
                    # Construct specific query
                    query_parts = [field] if field else []
                    if subfield:
                        query_parts.append(subfield)
                    if topics:
                        query_parts.extend(topics[:3])  # Limit to 3 topics for focus
                    
                    query = " ".join(query_parts)
                    logger.info(f"Using field-specific query: {query}")
                else:
                    # Handle simple string query
                    query = query_spec
                    logger.info(f"Using simple query: {query}")
                
                # Run iteration with this query
                result = self.run_iteration(query)
                
                # Extract papers_added from the result if it's a dictionary
                if isinstance(result, dict):
                    papers_added = result.get('papers_added', 0)
                else:
                    papers_added = 0
                    
                total_papers_processed += papers_added
                logger.info(f"Total papers processed: {total_papers_processed}")
                
                # Check if we've reached our processing target
                if total_papers_processed >= 10000000:
                    logger.info("Reached paper processing target. Completing research.")
                    break
            
            # Train model after processing all queries in this iteration
            if hasattr(self.learner, 'train'):
                logger.info("Training model with accumulated papers...")
                self.learner.train()
            
            # Generate a sample paper after each full iteration
            if i % 2 == 0:  # Every other iteration to reduce frequency
                self.generate_paper("Recent advances in AI research")
            
            # Pause between iterations
            if i < iterations - 1:  # Don't pause after the last iteration
                logger.info(f"Pausing between research iterations...")
                time.sleep(2)
        
        # Generate final research report
        logger.info("Generating final research report...")
        final_report = self._generate_research_report()
        
        return final_report

    def generate_paper(self, prompt):
        """Use the trained model to generate text based on the prompt"""
        # Check if we have a trained model
        if not hasattr(self.learner, 'models') or not self.learner.models or self.current_model not in self.learner.models:
            logger.warning(f"No trained model available for paper generation. Train the model first.")
            return "No model available to generate paper."
            
        # Get the current model from learner
        model = self.learner.models.get(self.current_model)
        if model is None:
            logger.warning(f"Current model {self.current_model} not found in learner models.")
            return "Current model not available."
            
        # This is a placeholder for actual text generation
        logger.info(f"Generating paper with prompt: {prompt}")
        generated_text = f"# AI Research Paper\n\n## Abstract\n\nThis is a placeholder for generated text based on: '{prompt}'\n\n"
        generated_text += f"Generated using model: {self.current_model}\n"
        
        self.save_generated_paper(generated_text)
        return generated_text

    def save_generated_paper(self, text):
        """Save the generated text as a file in the papers directory"""
        if not text:
            logger.warning("No text to save.")
            return
            
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_paper_{timestamp}.txt"
        
        try:
            # Ensure the papers directory exists
            papers_dir = CONFIG.get('papers_dir', 'papers')
            os.makedirs(papers_dir, exist_ok=True)
            
            # Save the text to file
            filepath = os.path.join(papers_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
                
            logger.info(f"Generated paper saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save generated paper: {e}")
            return None

    def _train_model(self):
        """Train with memory-efficient incremental learning"""
        # Create memory buffer of important samples
        memory_buffer = self._create_memory_buffer()
        
        # Train in batches with replay
        for epoch in range(CONFIG['training_epochs']):
            # Get current batch
            batch = self._get_training_batch()
            
            # Combine with memory samples
            combined_batch = batch + random.sample(memory_buffer, 
                                                 min(len(memory_buffer), len(batch)//2))
            
            # Train step
            loss = self.model.train_step(combined_batch)
            
            # Update memory buffer with important samples
            self._update_memory_buffer(batch, memory_buffer)
            
            # Clear gradients and empty cache
            self.model.zero_grad()
            torch.cuda.empty_cache()

    def _calculate_adaptive_learning_rate(self):
        """Calculate adaptive learning rate based on iteration and performance"""
        base_lr = CONFIG.get('base_learning_rate', 1e-4)
        
        # Adjust based on iteration number
        iteration_factor = max(0.1, 1.0 - (self.iteration / 100.0))
        
        # Adjust based on recent performance
        if self.performance_history:
            latest_f1 = self.performance_history[-1].get('overall_f1', 0.5)
            performance_factor = max(0.5, min(2.0, latest_f1 * 2))
        else:
            performance_factor = 1.0
            
        adaptive_lr = base_lr * iteration_factor * performance_factor
        logger.info(f"Using adaptive learning rate: {adaptive_lr:.2e}")
        return adaptive_lr
