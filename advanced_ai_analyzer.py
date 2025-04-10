import os
import re
import json
import time
import logging
import arxiv
import PyPDF2
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm
import pickle
from datetime import datetime
from collections import Counter, defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_research.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

# Global configuration
CONFIG = {
    # Data storage
    'papers_dir': 'papers',
    'models_dir': 'models',
    'knowledge_base_file': 'knowledge_base.json',
    'embeddings_file': 'paper_embeddings.pkl',
    'citation_network_file': 'citation_network.json',
    
    # Dataset parameters
    'max_papers_per_query': 100,  # Increased from 20
    'max_papers_total': 1000,    # Increased from 100
    'max_features': 2000,        # Maximum features for vectorizers
    
    # Training parameters
    'batch_size': 16,            # Reduced batch size for memory efficiency
    'learning_rate': 3e-5,       # Adjusted for better convergence
    'epochs': 30,                # Increased from 10
    'hidden_size': 512,          # Increased from 256
    'dropout_rate': 0.2,         # Adjusted for better generalization
    'max_seq_length': 512,       # Increased from 256
    'num_workers': 4,            # Increased from 2
    'random_seed': 42,
    'use_gpu': torch.cuda.is_available(),
    'use_mixed_precision': True, # Enable mixed precision for better GPU memory utilization
    'use_gradient_checkpointing': True, # Reduce memory footprint during training
    'progressive_loading': True, # Load data progressively instead of all at once
    'cache_embeddings': True,    # Cache embeddings to avoid recomputing
    'memory_efficient_backprop': True, # Use memory-efficient backpropagation
    'gradient_accumulation_steps': 4, # Simulate larger batch sizes with gradient accumulation
    
    # Advanced learning parameters
    'max_iterations': 20,        # Increased from 5
    'improvement_threshold': 0.005, # More sensitive to improvements
    'early_stopping_patience': 5,   # More patience for convergence
    'warmup_steps': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'save_interval': 2,
    
    # Self-improvement parameters
    'exploration_rate_decay': 0.95,  # For gradually reducing exploration
    'knowledge_retention_factor': 0.9, # For knowledge retention
    'concept_similarity_threshold': 0.75, # For concept clustering
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
random.seed(CONFIG['random_seed'])
if CONFIG['use_gpu']:
    torch.cuda.manual_seed_all(CONFIG['random_seed'])

# Create necessary directories
for directory in [CONFIG['papers_dir'], CONFIG['models_dir']]:
    os.makedirs(directory, exist_ok=True)

# Device configuration
device = torch.device('cuda' if CONFIG['use_gpu'] else 'cpu')
logger.info(f"Using device: {device}")
