CONFIG = {
    # Data collection
    'max_papers': 10000000,
    'ai_categories': ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML'],
    'papers_per_batch': 50, # Papers to download/process per batch
    'max_papers_per_query': 100, # Max papers from a single query
    'max_papers_total': 10000, # Target total papers in KB
    
    # Processing
    'initial_batch_size': 64,
    'min_batch_size': 8,
    'max_parallel_downloads': 16,
    'num_workers': 4, # Dataloader workers
    
    # Memory
    'max_concepts': 50000,
    'max_authors': 100000,
    'max_categories': 1000,
    'knowledge_base_file': 'kb.json', # Knowledge base save file
    'embeddings_file': 'embeddings.npz', # Embedding file (if applicable)
    'citation_network_file': 'citation_network.json',
    'papers_dir': 'papers',
    'models_dir': 'models',
    'streaming_cache_size': 1000,
    
    # Concept Extraction
    'max_features': 5000, # For TF-IDF vectorizers
    'top_n_concepts': 50, # Number of concepts to extract per paper
    'domain_keywords': ['machine learning', 'neural network', 'deep learning', 'artificial intelligence', 'data science'],

    # Embedding Generation (Using Sentence Transformers)
    'use_sentence_transformer': True,
    'sentence_transformer_model': 'all-MiniLM-L6-v2', # Efficient default model
    'embedding_size': 384, # Output dimension for all-MiniLM-L6-v2
    
    # Model Training (Classifier)
    'model_name': 'EnhancedClassifier', # 'EnhancedClassifier' or 'SimpleClassifier'
    'hidden_size': 768, # Hidden layer size for classifier
    'batch_size': 16, # Training batch size
    'epochs': 10,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 3,
    'use_mixed_precision': True,
    'gradient_accumulation_steps': 4,
    'max_grad_norm': 1.0,
    
    # Engine Settings
    'use_knowledge_graph': False, # KG features are experimental
    'adaptive_exploration': True,
    'knowledge_retention_score': 1.0, # Initial retention score
    'base_learning_rate': 1e-4 # For adaptive LR calculation
}
