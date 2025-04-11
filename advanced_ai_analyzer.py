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

# Import the canonical configuration
from config import CONFIG, DATA_DIR # Also import DATA_DIR if needed directly

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
