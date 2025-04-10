#!/usr/bin/env python3
"""
Advanced AI Research System
--------------------------
A recursively self-iterating AI researcher that learns from research papers
and continuously improves its understanding of AI and ML concepts.
"""

import os
import argparse
import logging
import time
import json
from datetime import datetime

from advanced_ai_analyzer import CONFIG, logger
from advanced_ai_analyzer_paper_processor import PaperProcessor
from advanced_ai_analyzer_knowledge_base import KnowledgeBase
from advanced_ai_analyzer_learning import LearningSystem
from advanced_ai_analyzer_engine import RecursiveResearchEngine

def setup_argparse():
    """Setup command line argument parsing"""
    parser = argparse.ArgumentParser(description='Advanced AI Research System')
    
    parser.add_argument('--query', type=str, default="artificial intelligence machine learning",
                        help='Initial research query')
    parser.add_argument('--iterations', type=int, default=CONFIG['max_iterations'],
                        help='Maximum number of research iterations')
    parser.add_argument('--papers', type=int, default=CONFIG['max_papers_per_query'],
                        help='Maximum papers to download per query')
    parser.add_argument('--papers-total', type=int, default=CONFIG['max_papers_total'],
                        help='Maximum total papers to download')
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'],
                        help='Batch size for model training')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--model', type=str, choices=['transformer', 'multitask'], default='transformer',
                        help='Model type to use')
    parser.add_argument('--papers-dir', type=str, default=CONFIG['papers_dir'],
                        help='Directory to store downloaded papers')
    parser.add_argument('--models-dir', type=str, default=CONFIG['models_dir'],
                        help='Directory to store trained models')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing models without training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def update_config(args):
    """Update configuration based on command line arguments"""
    CONFIG['max_papers_per_query'] = args.papers
    CONFIG['max_papers_total'] = args.papers_total
    CONFIG['batch_size'] = args.batch_size
    CONFIG['epochs'] = args.epochs
    CONFIG['papers_dir'] = args.papers_dir
    CONFIG['models_dir'] = args.models_dir
    
    # Create directories if they don't exist
    for directory in [CONFIG['papers_dir'], CONFIG['models_dir']]:
        os.makedirs(directory, exist_ok=True)
    
    # Update logging level if debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

def run_evaluation_only(args):
    """Run evaluation on existing models without training"""
    logger.info("Running evaluation only mode")
    
    # Initialize components
    kb = KnowledgeBase()
    learner = LearningSystem(kb)
    
    # Load and evaluate models
    for model_type in ['transformer', 'multitask']:
        if learner.load_model(model_type):
            logger.info(f"Evaluating {model_type} model...")
            eval_results = learner.evaluate(model_type=model_type)
            if eval_results:
                logger.info(f"Evaluation results for {model_type} model:")
                logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
                logger.info(f"  F1 Score: {eval_results['overall_f1']:.4f}")
            else:
                logger.warning(f"No evaluation results for {model_type} model")
        else:
            logger.warning(f"Could not load {model_type} model")
    
    # Generate learning curves
    learner.generate_learning_curves()

def print_welcome_message(args):
    """Print welcome message with configuration details"""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║             ADVANCED AI RESEARCH SYSTEM                      ║
    ║                                                              ║
    ║  A recursively self-iterating AI researcher that learns      ║
    ║  from research papers and continuously improves its          ║
    ║  understanding of AI and ML concepts.                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Initial query: {args.query}
    Max iterations: {args.iterations}
    Papers per query: {args.papers}
    Total papers limit: {args.papers_total}
    Using device: {CONFIG['use_gpu']}
    """)

def main():
    """Main function to run the enhanced AI research system with advanced learning capabilities"""
    # Parse command line arguments
    args = setup_argparse()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("ai_research.log"),
            logging.StreamHandler()
        ]
    )
    
    # Update configuration with enhanced settings
    update_config(args)
    
    # Set expanded dataset parameters - use more reasonable values for testing
    CONFIG['max_papers_per_query'] = min(20, args.papers)  # Use a smaller number for testing
    CONFIG['max_papers_total'] = min(100, args.papers_total)  # Use a smaller number for testing
    CONFIG['epochs'] = min(10, args.epochs)  # Use fewer epochs for testing
    CONFIG['hidden_size'] = 512  # Increase hidden size for more capacity
    CONFIG['max_iterations'] = min(5, args.iterations)  # Use fewer iterations for testing
    
    # Check if we should only run evaluation
    if args.eval_only:
        run_evaluation_only(args)
        return
    
    # Print welcome message with current configuration
    print_welcome_message(args)
    
    # Create necessary directories
    for directory in [args.papers_dir, args.models_dir, os.path.join(args.models_dir, 'reports'),
                     os.path.join(args.models_dir, 'concept_maps'), os.path.join(args.models_dir, 'frontiers')]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize the recursive research engine with advanced capabilities
    engine = RecursiveResearchEngine(
        papers_dir=args.papers_dir,
        models_dir=args.models_dir
    )
    
    # Configure the engine for advanced learning
    engine.adaptive_exploration = True  # Enable adaptive exploration
    engine.knowledge_retention_score = 1.0  # Initialize knowledge retention score
    
    # Start with a comprehensive query to gather diverse papers
    research_domains = [
        "reinforcement learning", "deep learning", "natural language processing",
        "computer vision", "graph neural networks", "generative models",
        "quantum machine learning", "federated learning", "self-supervised learning",
        "meta-learning", "neural architecture search", "explainable AI"
    ]
    
    # Use the provided query or a comprehensive one
    first_query = args.query if args.query else " ".join(research_domains[:3])
    
    # Run the engine for the specified number of iterations with continuous learning
    start_time = time.time()
    
    # First phase: Broad learning across domains (limited to 1-2 domains for testing)
    logger.info("Phase 1: Broad learning across multiple AI domains")
    for domain in research_domains[:min(2, len(research_domains))]:
        logger.info(f"Learning from domain: {domain}")
        try:
            engine.run_iteration(domain)
        except Exception as e:
            logger.error(f"Error processing domain {domain}: {e}")
        
    # Second phase: Focused learning based on identified research frontiers
    logger.info("Phase 2: Focused learning on identified research frontiers")
    final_report = engine.run(
        [first_query],  # Wrap in list for initial_queries parameter
        args.iterations
    )
    
    # Third phase: Self-improvement and knowledge consolidation
    logger.info("Phase 3: Self-improvement and knowledge consolidation")
    try:
        if hasattr(engine, 'research_frontiers') and engine.research_frontiers:
            # Focus on the most promising research frontiers
            # Note: research_frontiers is a list of strings (concept names), not dictionaries
            for frontier in engine.research_frontiers[:2]:  # Use top 2 frontiers
                logger.info(f"Consolidating knowledge in frontier: {frontier}")
                # Use the frontier concept as the query
                engine.run_iteration(frontier)
        else:
            logger.info("No research frontiers identified, skipping consolidation phase")
    except Exception as e:
        logger.error(f"Error in knowledge consolidation phase: {e}")
    
    end_time = time.time()
    
    # Print summary
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Generate comprehensive final report
    final_stats = {
        'total_iterations': engine.iteration,
        'total_papers': len(engine.kb.papers),
        'total_concepts': len(engine.kb.concept_index),
        'knowledge_retention': getattr(engine, 'knowledge_retention_score', 1.0),
        'research_frontiers': len(getattr(engine, 'research_frontiers', [])),
        'knowledge_gaps': len(getattr(engine, 'knowledge_gaps', set())),
        'execution_time': execution_time
    }
    
    # Save final statistics
    with open(os.path.join(args.models_dir, 'final_research_stats.json'), 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                ADVANCED RESEARCH COMPLETED                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Total iterations: {engine.iteration}
    Total papers processed: {len(engine.kb.papers)}
    Total concepts learned: {len(engine.kb.concept_index)}
    Research frontiers identified: {len(getattr(engine, 'research_frontiers', []))}
    Knowledge retention score: {getattr(engine, 'knowledge_retention_score', 1.0):.4f}
    Final F1 score: {final_report.get('evaluation', {}).get('f1_score', 0):.4f}
    Best performing model: {getattr(engine, 'current_model', 'simple')}
    
    Execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s
    
    Research reports saved in: {os.path.join(args.models_dir, 'reports')}
    Concept maps saved in: {os.path.join(args.models_dir, 'concept_maps')}
    Research frontiers saved in: {os.path.join(args.models_dir, 'frontiers')}
    """)

if __name__ == "__main__":
    main()
