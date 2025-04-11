#!/usr/bin/env python3
"""
Advanced AI Research System - Enhanced Mode
------------------------------------------
Runs the AI research system with all the advanced features enabled.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

from advanced_ai_analyzer import CONFIG, logger
from advanced_ai_analyzer_paper_processor import PaperProcessor
from advanced_ai_analyzer_knowledge_base import KnowledgeBase
from advanced_ai_analyzer_learning import LearningSystem
from advanced_ai_analyzer_engine import RecursiveResearchEngine
from utils.progress_display import ProgressTracker

def setup_argparse():
    """Setup command line argument parsing"""
    parser = argparse.ArgumentParser(description='Advanced AI Research System - Enhanced Mode')
    
    parser.add_argument('--query', type=str, default="reinforcement learning deep neural networks",
                      help='Initial research query')
    parser.add_argument('--iterations', type=int, default=5,
                      help='Maximum number of research iterations')
    parser.add_argument('--papers', type=int, default=30,
                      help='Maximum papers to download per query')
    parser.add_argument('--gpu', action='store_true',
                      help='Force GPU usage (even if not automatically detected)')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage (disable GPU)')
    parser.add_argument('--distributed', action='store_true',
                      help='Enable distributed training if multiple GPUs available')
    parser.add_argument('--incremental', action='store_true',
                      help='Enable incremental learning (default: True)')
    parser.add_argument('--knowledge-graph', action='store_true',
                      help='Enable knowledge graph generation (default: True)')
    parser.add_argument('--output-dir', type=str, default="research_output",
                      help='Directory to store output files')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    return parser.parse_args()

def update_config(args):
    """Update configuration based on command line arguments"""
    # Update GPU settings
    if args.gpu:
        CONFIG['use_gpu'] = True
    elif args.cpu:
        CONFIG['use_gpu'] = False
        
    # Update multi-GPU settings
    if args.distributed:
        CONFIG['use_distributed_training'] = True
    
    # Update learning settings
    CONFIG['use_incremental_learning'] = not args.gpu  # Enabled by default
    CONFIG['use_knowledge_graph'] = not args.knowledge_graph  # Enabled by default
    
    # Update paper limits
    CONFIG['max_papers_per_query'] = args.papers
    CONFIG['max_papers_total'] = args.papers * args.iterations * 2
    
    # Create output directories
    base_dir = args.output_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    papers_dir = os.path.join(run_dir, 'papers')
    models_dir = os.path.join(run_dir, 'models')
    reports_dir = os.path.join(models_dir, 'reports')
    
    CONFIG['papers_dir'] = papers_dir
    CONFIG['models_dir'] = models_dir
    
    # Create directories
    for directory in [run_dir, papers_dir, models_dir, reports_dir]:
        os.makedirs(directory, exist_ok=True)
        
    # Update logging level if debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
            
    logger.info(f"Updated configuration:")
    logger.info(f"  GPU Mode: {'Enabled' if CONFIG['use_gpu'] else 'Disabled'}")
    logger.info(f"  Multi-GPU: {'Enabled' if CONFIG['use_distributed_training'] else 'Disabled'}")
    logger.info(f"  Incremental Learning: {'Enabled' if CONFIG['use_incremental_learning'] else 'Disabled'}")
    logger.info(f"  Knowledge Graph: {'Enabled' if CONFIG['use_knowledge_graph'] else 'Disabled'}")
    logger.info(f"  Papers per query: {CONFIG['max_papers_per_query']}")
    logger.info(f"  Output directory: {run_dir}")

def print_welcome_message(args):
    """Print welcome message with configuration details"""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       ADVANCED AI RESEARCH SYSTEM - ENHANCED MODE            ║
    ║                                                              ║
    ║  A recursively self-iterating AI researcher with advanced    ║
    ║  transformer-based embeddings, knowledge graph capabilities, ║
    ║  and distributed training.                                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Initial query: {args.query}
    Iterations: {args.iterations}
    Papers per query: {args.papers}
    Running on: {"GPU" if CONFIG['use_gpu'] else "CPU"}
    """)

def main():
    """Main function to run the enhanced AI research system"""
    # Parse command line arguments
    args = setup_argparse()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("advanced_ai_research.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Update configuration with command line arguments
    update_config(args)
    
    # Print welcome message
    print_welcome_message(args)
    
    # Initialize the recursive research engine
    with ProgressTracker(total=100, desc="Initializing system", unit="steps") as progress:
        progress.update(10, metrics={"stage": "Creating paper processor"})
        processor = PaperProcessor(papers_dir=CONFIG['papers_dir'])
        
        progress.update(20, metrics={"stage": "Initializing knowledge base"})
        kb = KnowledgeBase()
        
        progress.update(20, metrics={"stage": "Initializing learning system"})
        learner = LearningSystem(kb, CONFIG['models_dir'])
        
        progress.update(20, metrics={"stage": "Creating research engine"})
        engine = RecursiveResearchEngine(
            papers_dir=CONFIG['papers_dir'],
            models_dir=CONFIG['models_dir']
        )
        
        progress.update(30, metrics={"stage": "Preparing for research"})
    
    # Run the research engine
    try:
        logger.info("Starting research process")
        start_time = time.time()
        
        # Run research with the initial query
        final_report = engine.run(
            [args.query],
            iterations=args.iterations
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Print completion message
        print(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                                                              ║
        ║              ADVANCED RESEARCH COMPLETED                     ║
        ║                                                              ║
        ╚══════════════════════════════════════════════════════════════╝
        
        Total iterations: {engine.iteration}
        Total papers processed: {len(engine.kb.papers)}
        Total concepts learned: {len(engine.kb.concept_index)}
        Research frontiers identified: {len(getattr(engine, 'research_frontiers', []))}
        Best performing model: {getattr(engine.model_trainer, 'current_model', 'unknown')}
        
        Execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s
        
        Reports saved in: {os.path.join(CONFIG['models_dir'], 'reports')}
        """)
        
    except KeyboardInterrupt:
        logger.info("Research process interrupted by user")
        print("\nResearch process interrupted. Saving current progress...")
        engine.kb.save()
        
    except Exception as e:
        logger.error(f"Error in research process: {e}", exc_info=True)
        print(f"\nError occurred during research process: {e}")
        
    finally:
        # Save the knowledge base
        engine.kb.save()
        logger.info("Knowledge base saved")

if __name__ == "__main__":
    main() 