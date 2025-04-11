"""
Transfer learning implementation for the AI Research System
"""

from advanced_ai_analyzer import *

def apply_transfer_learning(engine, query):
    """Apply transfer learning when no new papers are available"""
    logger.info(f"Applying transfer learning for query: '{query}'")
    
    # Check if sentence transformer is available
    if engine.processor.sentence_transformer is None:
        logger.warning("Sentence transformer not available in PaperProcessor. Skipping transfer learning.")
        return {'success': False, 'message': 'Sentence transformer unavailable'}
    
    # Find most similar papers to the query in our knowledge base
    # Convert query to embedding first
    query_embedding = engine.processor.sentence_transformer.encode(query)
    similar_papers = engine.kb.semantic_search(query_embedding, top_n=5)
    
    if not similar_papers:
        logger.warning("No similar papers found for transfer learning")
        return {'success': False, 'message': 'No similar papers found'}
    
    # Extract concepts from similar papers to enrich understanding
    transfer_concepts = set()
    for paper_id in similar_papers:
        if paper_id in engine.kb.papers and 'concepts' in engine.kb.papers[paper_id]:
            transfer_concepts.update(engine.kb.papers[paper_id]['concepts'])
    
    # Use these concepts to enhance the model's understanding
    if transfer_concepts:
        logger.info(f"Using {len(transfer_concepts)} concepts from similar papers for transfer learning")
        
        # Create a synthetic paper with these concepts for training
        synthetic_paper = {
            'id': f"synthetic_{int(time.time())}",
            'title': f"Transfer Learning for {query}",
            'abstract': " ".join(list(transfer_concepts)[:100]),
            'concepts': list(transfer_concepts),
            'embedding': None,  # Will be generated during processing
            'filepath': None,   # No actual file
            'authors': [],
            'categories': []
        }
        
        # Process the synthetic paper to generate embedding
        try:
            processed_paper = engine.processor._process_single_paper(synthetic_paper, extract_concepts=False)
            if processed_paper:
                # Add to knowledge base temporarily for this training session
                engine.kb.add_paper(processed_paper)
                logger.info("Added synthetic paper to knowledge base for transfer learning")
                
                # Train on the updated knowledge base
                logger.info(f"Training {engine.current_model} model with transfer learning...")
                
                # Calculate adaptive learning rate if method exists
                learning_rate = CONFIG['learning_rate']
                if hasattr(engine, '_calculate_adaptive_learning_rate'):
                    try:
                        learning_rate = engine._calculate_adaptive_learning_rate()
                    except:
                        pass
                
                engine.learner.train(model_type=engine.current_model, learning_rate=learning_rate)
                
                # Indicate success
                return {'success': True, 'message': 'Transfer learning applied successfully'}
        except Exception as e:
            logger.error(f"Error during transfer learning: {e}")
            return {'success': False, 'message': f'Error during transfer learning: {e}'}
        finally:
            # --- Cleanup --- 
            # Remove synthetic paper from KB regardless of training outcome
            if processed_paper and engine.kb.papers.get(processed_paper['id']): # Check if it was actually added
                # Use the new KB method to remove from metadata and FAISS
                engine.kb.remove_paper(processed_paper['id'])
                logger.info(f"Removed synthetic paper {processed_paper['id']} after transfer learning attempt.")
                # Consider saving KB state after removal
                engine.kb.save() # Save state after removal
    else:
        logger.warning("No concepts found for transfer learning")
        return {'success': False, 'message': 'No concepts found for transfer'}

    # Default return if no explicit success path hit (should ideally not happen)
    return {'success': False, 'message': 'Transfer learning did not complete successfully'}
