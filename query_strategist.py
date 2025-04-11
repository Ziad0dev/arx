import random
import numpy as np
from advanced_ai_analyzer import logger, CONFIG

class QueryStrategist:
    """Manages the strategy for generating the next research query."""

    def __init__(self, engine):
        """Initialize with a reference to the main engine or required components."""
        self.engine = engine # Needs access to kb, history, iteration, focus, gaps, frontiers
        self.exploration_rate = CONFIG.get('initial_exploration_rate', 0.3)

    def _enhance_query_for_ai_categories(self, query):
        """Enhance query to ONLY target core AI/ML categories from config."""
        ai_categories = CONFIG.get('ai_categories', [])
        if not ai_categories:
            logger.warning("No 'ai_categories' defined in config. Returning original query.")
            return query
        # Ensure query is not empty
        if not query.strip():
             query = "artificial intelligence" # Default if query is empty
        return f"({query}) AND ({' OR '.join(f'cat:{cat}' for cat in ai_categories)})"

    def generate_next_query(self, current_query):
        """Generate the next research query based on current state and strategy."""
        use_adaptive = getattr(self.engine, 'adaptive_exploration', True)
        iteration = self.engine.iteration
        research_focus = getattr(self.engine, 'research_focus', [])
        kb = self.engine.kb
        performance_history = getattr(self.engine, 'performance_history', [])
        research_frontiers = getattr(self.engine, 'research_frontiers', [])
        knowledge_gaps = getattr(self.engine, 'knowledge_gaps', set())

        if not use_adaptive or iteration < 2: # Use initial focus for first couple iterations
            if not research_focus:
                 next_query = "artificial intelligence" # Default if no focus
                 logger.warning("No initial research focus provided. Using default query.")
            else:
                # Cycle through initial queries if multiple provided
                query_idx = iteration % len(research_focus)
                focus = research_focus[query_idx]
                if isinstance(focus, dict):
                    field = focus.get('field', 'artificial intelligence')
                    subfield = focus.get('subfield', '')
                    topics = focus.get('topics', [])
                    query_parts = [field]
                    if subfield: query_parts.append(subfield)
                    if topics: query_parts.extend(topics[:2]) # Focus on top 2 topics
                    next_query = " ".join(query_parts)
                else:
                    next_query = str(focus) # Simple string query
            logger.info(f"Generating next query from initial focus: {next_query}")
            return self._enhance_query_for_ai_categories(next_query)

        # Adaptive strategy based on performance, frontiers, and gaps
        strategy_choice = random.random()

        # 1. Exploit worst performing areas (based on category F1 score)
        if strategy_choice < 0.4 and performance_history: # 40% chance
            latest_eval = performance_history[-1].get('metrics', {})
            # Check if 'per_category' exists and is a dict
            perf_by_cat = latest_eval.get('per_category')
            if isinstance(perf_by_cat, dict):
                 # Filter categories with sufficient support and valid F1 scores
                 valid_cats = []
                 for cat, metrics in perf_by_cat.items():
                      if isinstance(metrics, dict) and metrics.get('support', 0) >= 3 and 'f1' in metrics:
                           valid_cats.append((cat, metrics['f1']))

                 if valid_cats:
                      valid_cats.sort(key=lambda x: x[1]) # Sort by F1 asc
                      target_category = valid_cats[0][0]
                      top_concepts = kb.get_top_concepts(category=target_category, n=3)
                      next_query = f"{target_category} {' '.join(top_concepts)}"
                      logger.info(f"Generating next query to exploit low performance area: {target_category}")
                      return self._enhance_query_for_ai_categories(next_query)

        # 2. Explore research frontiers
        elif strategy_choice < 0.7 and research_frontiers: # 30% chance
            frontier_concept = random.choice(list(research_frontiers)) # Ensure it's a list/tuple
            related_concepts = kb.get_top_concepts(n=5) # Get some general related concepts
            if related_concepts: # Ensure related concepts are found
                 next_query = f"{frontier_concept} {random.choice(related_concepts)}"
            else:
                 next_query = frontier_concept # Fallback if no related concepts
            logger.info(f"Generating next query to explore research frontier: {frontier_concept}")
            return self._enhance_query_for_ai_categories(next_query)

        # 3. Explore knowledge gaps
        elif strategy_choice < 0.9 and knowledge_gaps: # 20% chance
             gap_term = random.choice(list(knowledge_gaps))
             related_concepts = kb.get_top_concepts(n=5)
             if related_concepts:
                 next_query = f"{gap_term} {random.choice(related_concepts)}"
             else:
                 next_query = gap_term # Fallback if no related concepts
             logger.info(f"Generating next query to explore knowledge gap: {gap_term}")
             return self._enhance_query_for_ai_categories(next_query)

        # 4. Default/Fallback: Refine current query or use top concepts (10% chance or if others fail)
        top_concepts = kb.get_top_concepts(n=5)
        if top_concepts:
             # Mix old field (first word of current query) with new concept
             current_field = current_query.split()[0] if current_query else "AI"
             next_query = f"{current_field} {random.choice(top_concepts)}"
        elif current_query:
             next_query = current_query # Fallback to repeating current query
        else:
             next_query = "artificial intelligence" # Absolute fallback

        logger.info(f"Generating next query using fallback strategy.")
        return self._enhance_query_for_ai_categories(next_query)

    def update_exploration_rate(self, eval_results):
         """Update exploration rate based on performance and decay."""
         base_decay = CONFIG.get('exploration_decay', 0.99)
         perf_factor = 1.0
         if eval_results and 'overall_f1' in eval_results:
              # Reduce exploration more if performance is high
              f1_score = eval_results['overall_f1']
              perf_factor = max(0.8, 1.0 - f1_score * 0.2) # Adjust impact

         self.exploration_rate *= (base_decay * perf_factor)
         self.exploration_rate = max(CONFIG.get('min_exploration_rate', 0.05), self.exploration_rate) # Use config floor
         logger.info(f"Updated exploration rate to: {self.exploration_rate:.3f}")
         # Optionally update the engine's exploration rate if needed elsewhere immediately
         # self.engine.exploration_rate = self.exploration_rate 