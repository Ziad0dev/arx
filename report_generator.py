import os
import json
from datetime import datetime
from advanced_ai_analyzer import logger, CONFIG

class ReportGenerator:
    """Handles the generation of research summary reports."""

    def __init__(self, engine):
        """Initialize with a reference to the main engine."""
        self.engine = engine # Need access to engine state (kb, history, etc.)
        self.reports_dir = os.path.join(CONFIG['models_dir'], 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_research_report(self, comprehensive=False):
        """Generate a comprehensive report summarizing the current research state.
        
        Args:
            comprehensive (bool): Whether to include more detailed information in the report
        """
        iteration = self.engine.iteration
        logger.info(f"Generating research report for iteration {iteration}")

        # --- Gather Data --- 
        # Performance Data
        latest_eval = None
        performance_trend = {'f1_change': 0, 'accuracy_change': 0}
        if self.engine.performance_history:
            latest_eval = self.engine.performance_history[-1]
            if len(self.engine.performance_history) > 1:
                prev_eval = self.engine.performance_history[-2]
                f1_trend = latest_eval.get('metrics', {}).get('overall_f1', 0) - prev_eval.get('metrics', {}).get('overall_f1', 0)
                accuracy_trend = latest_eval.get('metrics', {}).get('accuracy', 0) - prev_eval.get('metrics', {}).get('accuracy', 0)
                performance_trend = {'f1_change': f1_trend, 'accuracy_change': accuracy_trend}
        else:
            latest_eval = {'metrics': {'overall_f1': 0, 'accuracy': 0}, 'status': 'No evaluations yet'}

        # Knowledge Base Summary
        kb = self.engine.kb
        kb_summary = {
            'total_papers': len(kb.papers),
            'concepts': kb.count_concepts(),
            'categories': len(kb.category_index),
            'authors': len(kb.author_index),
            # Safely get latest published date
            'latest_paper_published': None
        }
        if kb.timestamp_index:
            # Assuming timestamp_index[0][0] is a timestamp or parsable date string
            latest_ts_info = kb.timestamp_index[0]
            # Handle potential format issues when displaying
            kb_summary['latest_paper_published'] = str(latest_ts_info[0]) 
            # Could add title: kb_summary['latest_paper_title'] = kb.papers.get(latest_ts_info[1], {}).get('title')
        
        # --- Create Report Dictionary --- 
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'iteration': iteration,
            'total_papers_in_kb': len(kb.papers),
            'knowledge_summary': kb_summary,
            'latest_performance': latest_eval, 
            'performance_trend': performance_trend,
            'recent_meta_learning_data': self.engine.meta_learning_data[-5:] if self.engine.meta_learning_data else [],
            'current_model_type': self.engine.model_trainer.current_model,
            'current_exploration_rate': self.engine.exploration_rate,
            'knowledge_retention_score': getattr(self.engine, 'knowledge_retention_score', None), # Safely get score
            'identified_knowledge_gaps': list(getattr(self.engine, 'knowledge_gaps', set()))[:20], # Show a sample
            'total_iterations': self.engine.iteration
        }
        
        # Add performance history
        if hasattr(self.engine, 'performance_history'):
            report['performance_history'] = self.engine.performance_history
            
            # Add final metrics if available
            if self.engine.performance_history:
                final_metrics = self.engine.performance_history[-1].get('metrics', {})
                report['final_accuracy'] = final_metrics.get('accuracy', 'N/A')
                report['final_f1_score'] = final_metrics.get('overall_f1', 'N/A')
        
        # Add comprehensive data if requested
        if comprehensive:
            # Add research frontiers
            if hasattr(self.engine, 'research_frontiers'):
                report['research_frontiers'] = self.engine.research_frontiers
            
            # Add concept clusters
            if hasattr(self.engine, 'knowledge_manager') and hasattr(self.engine.knowledge_manager, 'concept_clusters'):
                top_clusters = list(self.engine.knowledge_manager.concept_clusters.items())[:10]
                report['top_concept_clusters'] = {k: v for k, v in top_clusters}
            
            # Add more detailed paper summaries
            top_papers = []
            for paper_id, paper in list(kb.papers.items())[:20]:  # Limit to 20 papers
                top_papers.append({
                    'id': paper_id,
                    'title': paper.get('title', 'Unknown'),
                    'top_concepts': paper.get('concepts', [])[:5],
                    'published': paper.get('metadata', {}).get('published', 'Unknown')
                })
            report['top_papers'] = top_papers
            
            # Add execution statistics
            if hasattr(self.engine, 'execution_stats'):
                report['execution_stats'] = self.engine.execution_stats

        # --- Save Report --- 
        report_path = os.path.join(self.reports_dir, f'research_report_iter_{iteration}.json')
        try:
            temp_report_path = report_path + ".tmp"
            with open(temp_report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str) # Use default=str for non-serializable types
            os.replace(temp_report_path, report_path)
            logger.info(f"Generated research report: {report_path}")
            return report
        except Exception as e:
            logger.error(f"Failed to save research report: {e}")
            return None # Indicate failure 