from advanced_ai_analyzer import *
from advanced_ai_analyzer_paper_processor import PaperProcessor
from advanced_ai_analyzer_knowledge_base import KnowledgeBase
from advanced_ai_analyzer_learning import LearningSystem
from collections import Counter, defaultdict
import transfer_learning
import knowledge_graph
import transformer_models
from report_generator import ReportGenerator
from query_strategist import QueryStrategist
from knowledge_manager import KnowledgeManager
from model_trainer import ModelTrainer
import torch
import itertools
from utils.gpu_manager import GPUMonitor
import arxiv
from tqdm import tqdm
import json

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
        self.meta_learning_data = []
        self.research_trajectory = [] # Initialize trajectory tracking
        
        # Self-improvement parameters
        self.model_performance_by_domain = {}
        self.adaptive_exploration = True
        
        # Create directories for research artifacts
        self.reports_dir = os.path.join(models_dir, 'reports')
        self.concept_maps_dir = os.path.join(models_dir, 'concept_maps')
        self.research_frontiers_dir = os.path.join(models_dir, 'frontiers')
        
        for directory in [self.reports_dir, self.concept_maps_dir, self.research_frontiers_dir]:
            os.makedirs(directory, exist_ok=True)

        # Instantiate the reporter
        self.reporter = ReportGenerator(self)

        # Instantiate the query strategist
        self.query_strategist = QueryStrategist(self)
        self.knowledge_manager = KnowledgeManager(self)
        self.model_trainer = ModelTrainer(self)
        self.exploration_rate = self.query_strategist.exploration_rate
    
    def process_papers(self, query, max_results=1000):
        """Search ArXiv, download papers, and initiate processing."""
        logger.info(f"Searching ArXiv for query: '{query}' (max_results={max_results})")
        
        # --- Step 1: Search and Download from ArXiv --- 
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance # Or SubmittedDate
            )
            
            results = list(search.results())
            if not results:
                logger.warning(f"No papers found on ArXiv for query: {query}")
                return 0
            
            logger.info(f"Found {len(results)} potential papers on ArXiv.")
            
            papers_metadata = []
            download_count = 0
            # Ensure papers_dir exists (might be redundant if __init__ ensures it)
            os.makedirs(self.processor.papers_dir, exist_ok=True)

            for paper in tqdm(results, desc="Downloading PDFs"): 
                try:
                    # Extract ID: http://arxiv.org/abs/cs/9605103v1 -> cs/9605103
                    paper_id_raw = paper.entry_id.split('/abs/')[-1].split('v')[0]
                    # Sanitize paper ID for filename (replace / with _)
                    paper_id_safe = paper_id_raw.replace('/', '_') 
                    pdf_filename = f"{paper_id_safe}.pdf"
                    filepath = os.path.join(self.processor.papers_dir, pdf_filename)
                    
                    # Check if already exists (basic cache)
                    if not os.path.exists(filepath):
                        paper.download_pdf(dirpath=self.processor.papers_dir, filename=pdf_filename)
                        download_count += 1
                        time.sleep(0.5) # Be polite to ArXiv API
                    else:
                        logger.debug(f"PDF already exists: {filepath}")
                    
                    # Prepare metadata needed for processing batch
                    meta = {
                        'id': paper_id_raw, # Use the original ID for metadata key
                        'filepath': filepath,
                        'title': paper.title,
                        'abstract': paper.summary,
                        'authors': [str(a) for a in paper.authors],
                        'published': paper.published.isoformat(),
                        'updated': paper.updated.isoformat(),
                        'categories': paper.categories,
                        'pdf_url': paper.pdf_url,
                        'entry_id': paper.entry_id
                        # Add other fields if needed by KB or processor
                    }
                    papers_metadata.append(meta)
                    
                except Exception as download_err:
                    logger.warning(f"Failed to download or process metadata for {paper.entry_id}: {download_err}")
            
            logger.info(f"Successfully downloaded {download_count} new PDFs.")
            
        except Exception as search_err:
            logger.error(f"ArXiv search failed for query '{query}': {search_err}")
            return 0 # Return 0 papers processed on search failure
            
        # --- Step 2: Process Downloaded Papers Batch --- 
        if not papers_metadata:
            logger.warning("No valid papers metadata generated after download attempt. Skipping processing.")
            return 0
            
        # Pass the collected metadata to the processor
        # Keep the dynamic batch sizing logic for processing
        paper_stream = iter(papers_metadata)   
        processed_count = 0
        batch_size = 32 # Reset batch size for processing phase

        # The rest of the loop processing batches...
        while True:
            # Dynamically adjust batch size (consider removing if not needed)
            # batch_size = GPUMonitor.optimize_batch_size(batch_size)
            
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
            novelty = self.kb.concept_novelty_score(paper.get('concepts', []))
            
            score = 0.4*recency + 0.4*citations + 0.2*novelty
            scored_papers.append((score, paper))
            
        # Return top 20% highest scoring papers
        scored_papers.sort(reverse=True)
        return [p[1] for p in scored_papers[:int(len(scored_papers)*0.2)]]
    
    def _apply_transfer_learning(self, query):
        """Wrapper to call the decoupled transfer learning function."""
        # Pass the required components explicitly
        return transfer_learning.apply_transfer_learning(
            kb=self.kb,
            processor=self.processor,
            learner=self.learner,
            model_trainer=self.model_trainer,
            query=query
        )
    
    def _calculate_adaptive_batch_size(self):
        """Calculate adaptive batch size based on current knowledge and iteration"""
        base_size = CONFIG.get('papers_per_batch', 10)
        scaling_factor = min(2.0, 1.0 + (self.iteration / 10.0))
        knowledge_factor = max(1.0, self.knowledge_manager.knowledge_retention_score * 1.5)
        
        adaptive_size = int(base_size * scaling_factor * knowledge_factor)
        logger.info(f"Using adaptive batch size: {adaptive_size} papers")
        return adaptive_size
        
    def _generate_meta_learning_data(self, query, processed_count, eval_results):
        """Generate meta-learning data from research iteration"""
        current_model_type = self.model_trainer.current_model
        self.meta_learning_data.append({
            'iteration': self.iteration,
            'query': query,
            'papers_processed': processed_count,
            'model_type': current_model_type,
            'accuracy': eval_results.get('accuracy', 0.0),
            'f1_score': eval_results.get('overall_f1', 0.0),
            'exploration_rate': self.exploration_rate,
            'timestamp': time.time()
        })
    
    def run_iteration(self, query):
        """Run a single research iteration with the given query
        
        Args:
            query (str): Query to use for this iteration
            
        Returns:
            dict: Results of this iteration
        """
        logger.info(f"Running research iteration {self.iteration+1} with query: {query}")
        
        # Calculate adaptive batch size based on current state
        batch_size = self._calculate_adaptive_batch_size()
        
        # Record starting point (number of papers before this iteration)
        starting_paper_count = len(self.kb.papers)
        
        # Process papers for this query
        processed_count = self.process_papers(query, max_results=batch_size)
        
        # Get list of newly added papers (for incremental learning)
        if processed_count > 0:
            # Get the newest papers by ID, assuming they're the ones just added
            # This is a simplification, but should work for our purposes
            new_papers = list(self.kb.papers.keys())[-processed_count:]
        else:
            new_papers = []
            
        # Skip expensive training if no new papers were added
        if len(new_papers) == 0:
            logger.warning(f"No new papers were added from query: {query}")
            self.iteration += 1
            return {
                'query': query,
                'papers_processed': 0,
                'papers_added': 0,
                'status': 'no_new_papers'
            }
        
        # Use incremental learning if we already have a model
        if CONFIG.get('use_incremental_learning', True) and hasattr(self.model_trainer, 'current_model') and self.model_trainer.current_model:
            logger.info(f"Using incremental learning for {len(new_papers)} new papers")
            eval_results = self.model_trainer.incremental_train(new_papers, model_type=self.model_trainer.current_model)
        else:
            # Fall back to full retraining
            logger.info("No existing model, using full training")
            eval_results = self.model_trainer.train_and_evaluate()
        
        # Build knowledge graph with new papers
        if CONFIG.get('use_knowledge_graph', True):
            self.kb.extract_concept_relationships()
        
        # Update performance tracking
        if eval_results:
            self._track_domain_performance(query, eval_results)
            
            # Generate meta-learning data for this iteration
            self._generate_meta_learning_data(query, processed_count, eval_results)
            
            # Adaptive exploration rate adjustment
            if self.adaptive_exploration and 'accuracy' in eval_results:
                old_rate = self.exploration_rate
                # Decrease exploration if we're doing well, increase if we're doing poorly
                adjustment = 0.05 * (0.8 - eval_results['accuracy'])
                self.exploration_rate = max(0.1, min(0.5, self.exploration_rate + adjustment))
                logger.info(f"Adjusted exploration rate: {old_rate:.2f} -> {self.exploration_rate:.2f}")
        
        # Generate concept map periodically
        if self.iteration % 5 == 0:
            self._generate_concept_map()
        
        # Record trajectory step with results
        trajectory_step = {
            'iteration': self.iteration,
            'query': query,
            'papers_processed': processed_count,
            'papers_added': len(new_papers),
            'evaluation': eval_results if eval_results else {},
            'exploration_rate': self.exploration_rate,
            'timestamp': datetime.now().isoformat()
        }
        self.research_trajectory.append(trajectory_step)
        
        # Increment iteration counter
        self.iteration += 1
        
        return trajectory_step
    
    def build_knowledge_graph(self):
        """Build a knowledge graph from the current knowledge base
        
        Returns:
            Boolean indicating success
        """
        return self.knowledge_manager.update_knowledge_graph()
    
    def get_research_map(self):
        """Generate a research map based on current knowledge
        
        Returns:
            Dictionary with research map data
        """
        research_map = {
            'frontiers': self.knowledge_manager.research_frontiers,
            'clusters': [],
            'key_concepts': [],
            'citation_hubs': []
        }
        
        # Add concept clusters from manager
        if self.knowledge_manager.concept_clusters:
            for key, cluster in list(self.knowledge_manager.concept_clusters.items())[:5]:
                research_map['clusters'].append({
                    'name': key,
                    'concepts': cluster
                })
                
        # Add key concepts from knowledge graph (via manager)
        if self.knowledge_manager.graph_built:
            if hasattr(self.knowledge_manager.kg, 'get_concept_importance'):
                research_map['key_concepts'] = [item['concept'] for item in self.knowledge_manager.kg.get_concept_importance(limit=20)]
            
        # Add citation hubs (accessing kb directly is okay here)
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
        """Run the recursive research process
        
        Args:
            initial_queries (list): List of initial queries to start with
            iterations (int): Maximum number of iterations to run
            max_results (int): Maximum number of results per query
            
        Returns:
            dict: Final research report
        """
        logger.info(f"Starting recursive research with {len(initial_queries)} initial queries")
        
        # Initialize performance tracking
        self.performance_history = []
        self.research_trajectory = []
        
        # Use the first query as starting point
        current_query = initial_queries[0] if initial_queries else "artificial intelligence"
        
        # Initial exploration rate
        self.exploration_rate = CONFIG.get('exploration_rate', 0.3)
        
        # Save starting timestamp
        start_time = time.time()
        
        # Run the specified number of iterations
        for i in range(iterations):
            logger.info(f"\n=================================================")
            logger.info(f"RESEARCH ITERATION {i+1}/{iterations}")
            logger.info(f"=================================================")
            logger.info(f"Current query: {current_query}")
            
            # Run a single iteration
            iteration_result = self.run_iteration(current_query)
            
            # Generate report every few iterations
            if (i+1) % 3 == 0 or i == iterations - 1:
                logger.info(f"Generating research report at iteration {i+1}")
                report = self.reporter.generate_research_report()
                
                # Save report to disk
                report_path = os.path.join(self.reports_dir, f"report_iter_{i+1}.json")
                try:
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2)
                    logger.info(f"Saved research report to {report_path}")
                except Exception as e:
                    logger.error(f"Failed to save report: {e}")
            
            # Generate next query based on current knowledge
            if i < iterations - 1:  # Skip for last iteration
                # Use query strategist to generate next query
                next_query = self.query_strategist.generate_next_query(current_query)
                logger.info(f"Next query: {next_query}")
                current_query = next_query
            
            # Check if we've reached the target paper count
            if len(self.kb.papers) >= CONFIG.get('max_papers_total', 10000):
                logger.info(f"Reached target paper count: {len(self.kb.papers)}")
                break
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Generate final research map
        logger.info("Generating final research map")
        research_map = self.get_research_map()
        
        # Save research map
        map_path = os.path.join(self.reports_dir, "final_research_map.json")
        try:
            with open(map_path, 'w') as f:
                json.dump(research_map, f, indent=2)
            logger.info(f"Saved final research map to {map_path}")
        except Exception as e:
            logger.error(f"Failed to save research map: {e}")
        
        # Extract research frontiers
        if CONFIG.get('use_knowledge_graph', True):
            logger.info("Extracting research frontiers from knowledge graph")
            self.research_frontiers = self.knowledge_manager.identify_research_frontiers()
            
            # Save frontiers
            frontiers_path = os.path.join(self.research_frontiers_dir, "research_frontiers.json")
            try:
                with open(frontiers_path, 'w') as f:
                    json.dump(self.research_frontiers, f, indent=2)
                logger.info(f"Saved research frontiers to {frontiers_path}")
            except Exception as e:
                logger.error(f"Failed to save research frontiers: {e}")
        
        # Generate comprehensive final report
        logger.info("Generating final research report")
        final_report = self.reporter.generate_research_report(comprehensive=True)
        
        # Log performance metrics
        if final_report.get('evaluation'):
            logger.info(f"Final model performance:")
            for metric, value in final_report['evaluation'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        
        # Generate summary statistics
        final_stats = {
            'total_iterations': self.iteration,
            'total_papers': len(self.kb.papers),
            'total_concepts': len(self.kb.concept_index),
            'knowledge_retention': getattr(self.knowledge_manager, 'knowledge_retention_score', 1.0),
            'research_frontiers': len(getattr(self, 'research_frontiers', [])),
            'execution_time': execution_time,
            'execution_time_formatted': f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        }
        
        # Add to final report
        final_report['statistics'] = final_stats
        
        # Save final report
        final_report_path = os.path.join(self.reports_dir, "final_research_report.json")
        try:
            with open(final_report_path, 'w') as f:
                json.dump(final_report, f, indent=2)
            logger.info(f"Saved final research report to {final_report_path}")
        except Exception as e:
            logger.error(f"Failed to save final report: {e}")
        
        # Save the knowledge base
        self.kb.save()
        
        # Print summary
        logger.info("\n=================================================")
        logger.info(f"RESEARCH COMPLETE")
        logger.info(f"=================================================")
        logger.info(f"Total iterations: {self.iteration}")
        logger.info(f"Total papers processed: {len(self.kb.papers)}")
        logger.info(f"Total concepts learned: {len(self.kb.concept_index)}")
        logger.info(f"Research frontiers identified: {len(getattr(self, 'research_frontiers', []))}")
        logger.info(f"Execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        return final_report

    def generate_paper(self, prompt):
        """Use the trained model to generate text based on the prompt"""
        # Use model trainer's current model
        current_model_type = self.model_trainer.current_model
        if not hasattr(self.learner, 'models') or not self.learner.models or current_model_type not in self.learner.models:
            logger.warning(f"No trained model '{current_model_type}' available...")
            return "No model available..."

        model = self.learner.models.get(current_model_type)
        if model is None:
            logger.warning(f"Current model '{current_model_type}' not found...")
            return "Current model not available."
            
        logger.info(f"Generating paper with prompt: {prompt} using model {current_model_type}")
        # Placeholder generation logic
        generated_text = f"# AI Research Paper ({current_model_type})\n\n## Abstract\n\nPlaceholder... '{prompt}'\n"
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

    def _attempt_model_improvement(self):
        logger.info("Checking for potential model architecture improvements...")
        # Could call a method on ModelTrainer if logic moves there
        pass

    def _track_domain_performance(self, query, eval_results):
        logger.debug("Tracking domain performance (Placeholder)")
        pass

    def _generate_concept_map(self):
        logger.debug("Generating concept map (Placeholder)")
        pass
