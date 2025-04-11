import os
import torch
import multiprocessing

# Define base directory for data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True) # Ensure the data directory exists

# GPU Device and Count Configuration
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_GPU = GPU_COUNT > 0
CURRENT_DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

# Auto-configure based on hardware
CPU_COUNT = multiprocessing.cpu_count()
RECOMMENDED_WORKERS = max(1, min(CPU_COUNT - 1, 8))  # Leave one CPU free, max 8

CONFIG = {
    # Environment / Global
    'random_seed': 42,
    'use_gpu': USE_GPU,
    'gpu_count': GPU_COUNT,
    'current_device': CURRENT_DEVICE,
    
    # Database configuration
    'use_database': True,
    'mongodb_uri': os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/'),
    'mongodb_db': 'ai_analyzer',
    
    # Data collection
    'max_papers': 10000000,
    'ai_categories': ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML'],
    'papers_per_batch': 100,  # Increased from 50
    'max_papers_per_query': 200,  # Increased from 100
    'max_papers_total': 20000,  # Increased from 10000
    
    # Processing
    'initial_batch_size': 64,
    'min_batch_size': 8,
    'max_parallel_downloads': 24,  # Increased from 16
    'num_workers': RECOMMENDED_WORKERS,
    'progressive_loading': True,
    'cache_embeddings': True,
    
    # Memory
    'max_concepts': 100000,  # Increased from 50000
    'max_authors': 150000,  # Increased from 100000
    'max_categories': 1500,  # Increased from 1000
    'knowledge_base_file': os.path.join(DATA_DIR, 'knowledge_base.json'),
    'embeddings_file': os.path.join(DATA_DIR, 'embeddings.pkl'),
    'citation_network_file': 'citation_network.json',
    'papers_dir': 'papers',
    'models_dir': 'models',
    'streaming_cache_size': 2000,  # Increased from 1000
    'faiss_index_file': os.path.join(DATA_DIR, 'kb_vector.index'),
    
    # Advanced Embedding Management
    'embedding_cache_dir': os.path.join(DATA_DIR, 'embeddings'),
    'embedding_cache_size': 20000,  # Increased from 10000
    'embedding_batch_size': 16,  # Increased from 8
    'embedding_max_length': 768,  # Increased from 512
    'embedding_max_workers': RECOMMENDED_WORKERS,
    
    # Transformer Models
    'embedding_model': 'allenai/specter2',
    'embedding_size': 768,
    'use_sentence_transformer': True,
    'sentence_transformer_model': 'sentence-transformers/all-mpnet-base-v2',
    
    # Concept Extraction
    'max_features': 7500,  # Increased from 5000
    'top_n_concepts': 75,  # Increased from 50
    'domain_keywords': ['machine learning', 'neural network', 'deep learning', 'artificial intelligence', 'data science'],

    # Model Training Parameters
    'model_name': 'EnhancedClassifier',
    'hidden_size': 1024,  # Increased from 768
    'dropout_rate': 0.2,
    'max_seq_length': 768,  # Increased from 512
    'batch_size': 32 if GPU_COUNT > 0 else 8,
    'epochs': 15,  # Increased from 10
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 0.1,
    'early_stopping_patience': 4,  # Increased from 3
    
    # Advanced Training Features
    'use_mixed_precision': True,
    'use_gradient_checkpointing': True,
    'memory_efficient_backprop': True,
    'gradient_accumulation_steps': 4,
    'max_grad_norm': 1.0,
    'save_interval': 2,
    
    # Multi-GPU Training
    'use_distributed_training': GPU_COUNT > 1,
    'distributed_backend': 'nccl' if USE_GPU else 'gloo',
    'distributed_world_size': GPU_COUNT if GPU_COUNT > 0 else 1,
    
    # Incremental Learning
    'use_incremental_learning': True,
    'replay_buffer_size': 2000,  # Increased from 1000
    'incremental_learning_rate': 5e-5,
    'incremental_epochs': 5,  # Increased from 3
    
    # Knowledge Graph
    'use_knowledge_graph': True,
    'neo4j_uri': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
    'neo4j_user': os.environ.get('NEO4J_USER', 'neo4j'),
    'neo4j_password': os.environ.get('NEO4J_PASSWORD', 'password'),
    'relation_confidence_threshold': 0.45,  # Lowered from 0.5
    'max_relationships_per_concept': 200,  # Increased from 100
    
    # Engine & Advanced Learning Settings
    'max_iterations': 30,  # Increased from 20
    'improvement_threshold': 0.003,  # Lowered from 0.005
    'adaptive_exploration': True,
    'exploration_rate_decay': 0.92,  # Changed from 0.95
    'knowledge_retention_factor': 0.92,  # Increased from 0.9
    'knowledge_retention_score': 1.0,
    'concept_similarity_threshold': 0.7,  # Lowered from 0.75 for broader concept matching
    'base_learning_rate': 1e-4
}

# Create embedding cache directory if using caching
if CONFIG['cache_embeddings'] and CONFIG['embedding_cache_dir']:
    os.makedirs(CONFIG['embedding_cache_dir'], exist_ok=True)
