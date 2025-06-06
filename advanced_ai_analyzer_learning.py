from advanced_ai_analyzer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import logging
import math
import uuid
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from advanced_ai_analyzer import CONFIG
from collections import defaultdict
import copy
from transformers import AutoModel, AutoTokenizer, AutoConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

class PaperDataset(Dataset):
    """Dataset for paper embeddings and their categories with memory-efficient options"""
    def __init__(self, embeddings, labels, use_memmap=False, memmap_dir=None):
        self.use_memmap = use_memmap
        self.memmap_dir = memmap_dir
        self.cleanup_needed = False
        
        if use_memmap and memmap_dir:
            # Create directory if it doesn't exist
            os.makedirs(memmap_dir, exist_ok=True)
            
            # Create memory-mapped arrays
            embedding_path = os.path.join(memmap_dir, f'embeddings_{uuid.uuid4().hex}.dat')
            labels_path = os.path.join(memmap_dir, f'labels_{uuid.uuid4().hex}.dat')
            
            # Save shape info for reconstruction
            self.embedding_shape = embeddings.shape
            
            # Create memory-mapped arrays
            self.embeddings = np.memmap(embedding_path, dtype='float32', mode='w+', shape=embeddings.shape)
            self.labels = np.memmap(labels_path, dtype='int32', mode='w+', shape=labels.shape)
            
            # Copy data to memory-mapped arrays
            self.embeddings[:] = embeddings[:]
            self.labels[:] = labels[:]
            self.embeddings.flush()
            self.labels.flush()
            
            # Store paths for cleanup
            self.embedding_path = embedding_path
            self.labels_path = labels_path
            self.cleanup_needed = True
            
            logger.info(f"Created memory-mapped dataset with {len(embeddings)} samples")
        else:
            # Use in-memory arrays
            self.embeddings = embeddings
            self.labels = labels
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return {
            'embedding': torch.FloatTensor(self.embeddings[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def cleanup(self):
        """Clean up memory-mapped files when done"""
        if self.cleanup_needed:
            # Close memmap files
            if hasattr(self, 'embeddings') and isinstance(self.embeddings, np.memmap):
                try:
                    self.embeddings._mmap.close()
                except:
                    pass
                
            # Delete the file
            try:
                os.remove(self.memmap_path)
                logger.info(f"Removed memory-mapped file {self.memmap_path}")
            except Exception as e:
                logger.warning(f"Failed to remove memory-mapped file: {e}")
                
    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup()


class LRUCache:
    """Least Recently Used Cache for efficient memory management"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def __getitem__(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def __contains__(self, key):
        return key in self.cache


class StreamingPaperDataset(Dataset):
    """Memory-efficient dataset that loads papers from the knowledge base on demand
    
    This dataset doesn't load all embeddings into memory at once, but instead
    generates them on-the-fly as needed, which is crucial for very large datasets.
    """
    def __init__(self, knowledge_base, categories):
        """Initialize a streaming dataset from the knowledge base
        
        Args:
            knowledge_base: The knowledge base containing papers
            categories: List of all possible categories
        """
        self.kb = knowledge_base
        self.categories = categories
        
        # Create mapping from categories to indices
        self.cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        # Create list of paper IDs for indexing
        self.paper_ids = list(self.kb.papers.keys())
        
        # Create transformer for embedding generation
        self.transformer = None
        if CONFIG.get('use_sentence_transformer', False):
            try:
                from sentence_transformers import SentenceTransformer
                self.transformer = SentenceTransformer(CONFIG.get('sentence_transformer_model', 'all-MiniLM-L6-v2'))
                logger.info(f"Initialized SentenceTransformer for streaming dataset")
            except ImportError:
                logger.warning("Could not import SentenceTransformer, falling back to TF-IDF")
        
        # Create TF-IDF vectorizer for fallback
        self.vectorizer = TfidfVectorizer(max_features=CONFIG['embedding_size'])
        
        # Fit vectorizer on a sample of documents to establish vocabulary
        sample_docs = []
        sample_size = min(1000, len(self.paper_ids))
        if self.paper_ids:
            for paper_id in random.sample(self.paper_ids, min(sample_size, len(self.paper_ids))):
                paper = self.kb.papers[paper_id]
                if 'title' in paper and 'abstract' in paper:
                    sample_docs.append(f"{paper['title']} {paper['abstract']}")
        
        if sample_docs:
            self.vectorizer.fit(sample_docs)
            logger.info(f"Fitted TF-IDF vectorizer on {len(sample_docs)} sample documents")
        
        # Initialize cache for frequently accessed embeddings
        self.embedding_cache = LRUCache(CONFIG.get('streaming_cache_size', 1000))
        
        logger.info(f"Created streaming dataset with {len(self.paper_ids)} papers and {len(categories)} categories")
    
    def __len__(self):
        return len(self.paper_ids)
    
    def __getitem__(self, idx):
        paper_id = self.paper_ids[idx]
        
        # Check cache first
        if paper_id in self.embedding_cache:
            embedding, label = self.embedding_cache[paper_id]
            return {
                'embedding': torch.FloatTensor(embedding),
                'label': torch.tensor(label, dtype=torch.long)
            }
        
        # Get paper data
        paper = self.kb.papers[paper_id]
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        # Generate embedding
        if self.transformer is not None and (title or abstract):
            # Use sentence transformer if available
            text = f"{title} {abstract}"
            embedding = self.transformer.encode(text)
            
            # Resize if needed
            if len(embedding) > CONFIG['embedding_size']:
                embedding = embedding[:CONFIG['embedding_size']]
            elif len(embedding) < CONFIG['embedding_size']:
                embedding = np.pad(embedding, (0, CONFIG['embedding_size'] - len(embedding)))
        else:
            # Fallback to TF-IDF
            text = f"{title} {abstract}"
            if not text.strip():
                # If no text, use a random embedding
                embedding = np.random.randn(CONFIG['embedding_size'])
            else:
                try:
                    tfidf = self.vectorizer.transform([text])
                    embedding = tfidf.toarray()[0]
                    
                    # Resize if needed
                    if len(embedding) > CONFIG['embedding_size']:
                        embedding = embedding[:CONFIG['embedding_size']]
                    elif len(embedding) < CONFIG['embedding_size']:
                        embedding = np.pad(embedding, (0, CONFIG['embedding_size'] - len(embedding)))
                except:
                    # If vectorization fails, use a random embedding
                    embedding = np.random.randn(CONFIG['embedding_size'])
        
        # Get label (use first category or default to 0)
        if 'metadata' in paper and 'categories' in paper['metadata'] and paper['metadata']['categories']:
            for cat in paper['metadata']['categories']:
                if cat in self.cat_to_idx:
                    label = self.cat_to_idx[cat]
                    break
            else:
                label = 0
        else:
            label = 0
        
        # Add to cache
        self.embedding_cache[paper_id] = (embedding, label)
        
        return {
            'embedding': torch.FloatTensor(embedding),
            'label': torch.tensor(label, dtype=torch.long)
        }


class SelfAttention(nn.Module):
    """Self-attention mechanism for capturing dependencies in embeddings"""
    
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, hidden_size)
        query = self.query(x).unsqueeze(1)  # (batch_size, 1, hidden_size)
        key = self.key(x).unsqueeze(1)      # (batch_size, 1, hidden_size)
        value = self.value(x).unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Scaled dot-product attention
        scores = torch.bmm(query, key.transpose(1, 2)) / self.scale  # (batch_size, 1, 1)
        attention = torch.softmax(scores, dim=2)
        context = torch.bmm(attention, value).squeeze(1)  # (batch_size, hidden_size)
        
        return context

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, x):
        residual = x
        out = self.layers(x)
        return out + residual  # Skip connection

class EnhancedClassifier(nn.Module):
    """Enhanced neural network classifier with attention and residual connections"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(EnhancedClassifier, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.attention = SelfAttention(hidden_size)
        
        # Multiple residual blocks for deep learning
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(3)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        # Project input to hidden dimension
        x = self.input_projection(x)
        x = nn.functional.relu(x)
        
        # Apply attention mechanism
        x = self.attention(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
            
        # Classification layer
        return self.classifier(x)

class SimpleClassifier(nn.Module):
    """Advanced classifier with multi-head attention for paper analysis"""
    
    def __init__(self, input_size, hidden_size, num_categories, num_heads=4):
        super(SimpleClassifier, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Multi-head attention mechanism
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),  # Using GELU activation (from transformers)
            nn.Dropout(CONFIG['dropout_rate']),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Output layer
        self.classifier = nn.Linear(hidden_size, num_categories)
        
    def split_heads(self, x, batch_size):
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        x = x.view(batch_size, 1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, 1, head_dim]
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_projection(x)
        residual = x
        
        # Multi-head attention
        q = self.split_heads(self.query_proj(x), batch_size)
        k = self.split_heads(self.key_proj(x), batch_size)
        v = self.split_heads(self.value_proj(x), batch_size)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, 1, self.num_heads * self.head_dim).squeeze(1)
        
        # Output projection
        attn_output = self.output_proj(context)
        
        # Add & Norm
        x = self.norm1(attn_output + residual)
        
        # Feed-forward network
        residual = x
        x = self.ffn(x)
        
        # Add & Norm
        x = self.norm2(x + residual)
        
        # Classification
        return self.classifier(x)

class LearningSystem:
    """Advanced learning system with state-of-the-art models"""
    
    def __init__(self, knowledge_base, models_dir=CONFIG['models_dir']):
        self.kb = knowledge_base
        self.models_dir = models_dir
        self.current_model = None
        self.categories = []  # Store categories for classification
        self.cat_to_idx = {}
        self.idx_to_cat = {}
        self.training_history = {} # Moved initialization here
        self.iterations = 0 # Initialize iteration counter
        self.concept_encoder = None # For encoding concepts
        self.embeddings = None
        self.labels = None
        self.label_map = {}
        
        # Initialize models dictionary
        self.models = {}
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Prepare initial data
        self.prepare_data()
        
        # Initialize concept encoder
        self._initialize_concept_encoder()
    
    def _initialize_concept_encoder(self):
        """Initialize concept encoder (simplified version)"""
        # In this simplified version, we don't use a separate concept encoder
        # We'll just use TF-IDF for concept representation
        self.concept_encoder = None
        logger.info("Using simplified concept encoding with TF-IDF")
    
    def _prepare_training_data(self):
        """Prepare training data embeddings and labels"""
        logger.info("Preparing training data...")
        if not self.kb.papers:
            logger.error("Knowledge base paper metadata store (kb.papers) is empty. Cannot prepare training data.")
            return False

        embeddings = []
        category_indices = []
        # Create category mapping from the KB's category index
        self.categories = sorted(list(self.kb.category_index.keys()))
        if not self.categories:
            logger.error("No categories found in the knowledge base category index. Cannot assign labels.")
            return False
        category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        logger.info(f"Using {len(self.categories)} categories for training: {self.categories}")

        # Iterate through papers in KB metadata store
        paper_ids = list(self.kb.papers.keys())
        logger.info(f"Total papers in KB metadata to consider: {len(paper_ids)}")

        # Filter papers based on category and embedding availability
        valid_paper_count = 0
        missing_embedding_count = 0
        missing_category_count = 0
        no_valid_category_count = 0

        for paper_id in paper_ids:
            # 1. Get metadata directly from kb.papers
            paper_meta = self.kb.papers.get(paper_id)
            if not paper_meta:
                logger.warning(f"Metadata missing for paper_id {paper_id} in kb.papers. Skipping.")
                continue

            # 2. Get embedding using kb.get_embedding
            paper_embedding = self.kb.get_embedding(paper_id)

            # Check 2a: Does it have a valid embedding?
            if paper_embedding is None:
                # logger.debug(f"Paper {paper_id} skipped: Missing embedding.") # Debug level can be noisy
                missing_embedding_count += 1
                continue
            if len(paper_embedding) != self.kb.embedding_dim:
                # logger.debug(f"Paper {paper_id} skipped: Embedding dim mismatch ({len(paper_embedding)} vs {self.kb.embedding_dim}).")
                missing_embedding_count += 1
                continue

            # 3. Check categories from metadata
            paper_categories = paper_meta.get('categories', [])
            if not paper_categories:
                # logger.debug(f"Paper {paper_id} skipped: No categories listed in metadata.")
                missing_category_count += 1
                continue
                
            # Check 3a: Does it have at least one category known to the KB index?
            valid_categories_in_paper = [cat for cat in paper_categories if cat in self.categories]
            if not valid_categories_in_paper:
                # logger.debug(f"Paper {paper_id} skipped: Categories {paper_categories} not in known training categories.")
                no_valid_category_count += 1
                continue

            # If valid, add embedding and ONE category index (first valid one)
            embeddings.append(paper_embedding)
            category_indices.append(category_to_idx[valid_categories_in_paper[0]])
            valid_paper_count += 1

        # Log summary statistics
        logger.info(f"Finished filtering. Valid papers for training: {valid_paper_count}")
        logger.info(f"Papers skipped due to missing/invalid embedding: {missing_embedding_count}")
        logger.info(f"Papers skipped due to no categories in metadata: {missing_category_count}")
        logger.info(f"Papers skipped due to no known/valid categories: {no_valid_category_count}")

        if not embeddings:
            logger.error("No valid training data after filtering")
            return False

        self.X = np.array(embeddings)
        self.y = np.array(category_indices)
        logger.info(f"Prepared training data with {len(self.X)} samples.")
        return True

    def prepare_data(self):
        """Prepare and split data for training"""
        if not self._prepare_training_data():
            return None, None, None, None
            
        # Split data with fallback for small datasets
        try:
            # Check if we have enough samples for stratified split
            unique_classes, class_counts = np.unique(self.y, return_counts=True)
            min_samples_per_class = np.min(class_counts)
            
            # Need at least 2 samples per class for stratified split
            if len(self.X) > 10 and len(unique_classes) >= 2 and min_samples_per_class >= 3:
                logger.info(f"Using stratified split with {len(unique_classes)} classes (min {min_samples_per_class} samples per class)")
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X, self.y, test_size=0.2, stratify=self.y, random_state=CONFIG.get('random_seed', 42)
                )
            else:
                # For small datasets or imbalanced data, use simple random split
                logger.warning("Using random split without stratification (small dataset or imbalanced classes)")
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X, self.y, test_size=0.2, stratify=None, random_state=CONFIG.get('random_seed', 42)
                )
                
                # If still too small, use all data for both training and validation
                if len(X_train) < 5 or len(X_val) < 2:
                    logger.warning("Dataset too small, using all data for both training and validation")
                    X_train, y_train = self.X, self.y
                    X_val, y_val = self.X.copy(), self.y.copy()
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            logger.warning("Falling back to using all data for both training and validation")
            # Use all data as both training and validation in case of any error
            X_train, y_train = self.X, self.y
            X_val, y_val = self.X.copy(), self.y.copy()
            
        logger.info(f"Data split complete: {len(X_train)} training samples, {len(X_val)} validation samples")
        return X_train, X_val, y_train, y_val
    
    def train(self, model_type='enhanced', learning_rate=1e-4):
        """Train model with robust data validation"""
        if not self._prepare_training_data():
            logger.error("No valid training data available")
            return
            
        if self.X is None or self.y is None:
            logger.error("Training data not initialized")
            return
            
        # Split data with fallback for small datasets
        try:
            # Check if we have enough samples for stratified split
            unique_classes, class_counts = np.unique(self.y, return_counts=True)
            min_samples_per_class = np.min(class_counts)
            
            # Need at least 2 samples per class for stratified split
            if len(self.X) > 10 and len(unique_classes) >= 2 and min_samples_per_class >= 3:
                logger.info(f"Using stratified split with {len(unique_classes)} classes (min {min_samples_per_class} samples per class)")
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X, self.y, test_size=0.2, stratify=self.y, random_state=CONFIG.get('random_seed', 42)
                )
            else:
                # For small datasets or imbalanced data, use simple random split
                logger.warning("Using random split without stratification (small dataset or imbalanced classes)")
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X, self.y, test_size=0.2, stratify=None, random_state=CONFIG.get('random_seed', 42)
                )
                
                # If still too small, use all data for both training and validation
                if len(X_train) < 5 or len(X_val) < 2:
                    logger.warning("Dataset too small, using all data for both training and validation")
                    X_train, y_train = self.X, self.y
                    X_val, y_val = self.X.copy(), self.y.copy()
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            logger.warning("Falling back to using all data for both training and validation")
            # Use all data as both training and validation in case of any error
            X_train, y_train = self.X, self.y
            X_val, y_val = self.X.copy(), self.y.copy()
            
        logger.info(f"Data split complete: {len(X_train)} training samples, {len(X_val)} validation samples")
        
        # Create datasets
        train_dataset = PaperDataset(X_train, y_train)
        val_dataset = PaperDataset(X_val, y_val)
        
        # Create dataloaders
        batch_size = min(CONFIG['batch_size'], len(X_train))
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=CONFIG['num_workers']
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=CONFIG['num_workers']
        )
        
        # Set input size and num_categories
        input_size = self.X.shape[1]
        num_categories = len(self.categories)
        
        # Initialize model
        hidden_size = CONFIG['hidden_size']
        
        if model_type == 'enhanced':
            model = EnhancedClassifier(input_size, hidden_size, num_categories)
        elif model_type == 'simple':
            model = SimpleClassifier(input_size, hidden_size, num_categories)
        else:
            logger.warning(f"Unknown model type: {model_type}, using enhanced model")
            model = EnhancedClassifier(input_size, hidden_size, num_categories)
        
        # Enable gradient checkpointing for memory efficiency if supported and configured
        if CONFIG.get('use_gradient_checkpointing', False) and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for memory efficiency")
        
        model = model.to(device)
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=CONFIG['weight_decay']
        )
        
        # Use a one-cycle learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,  # Peak learning rate
            epochs=CONFIG['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Spend 30% of time warming up
            div_factor=25,  # Initial LR = max_lr/div_factor
            final_div_factor=1000,  # Final LR = max_lr/final_div_factor
            anneal_strategy='cos'  # Cosine annealing
        )
        
        # Initialize loss functions
        category_criterion = nn.CrossEntropyLoss()
        
        # Initialize variables to track training progress
        best_val_loss = float('inf')
        best_accuracy = 0.0 # Initialize best accuracy
        patience = CONFIG['early_stopping_patience']
        patience_counter = 0
        best_model_state = None
        
        # Initialize training history
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'f1_score': [],
            'iterations': []
        }
        
        logger.info(f"Starting training with {len(train_loader.dataset)} samples ({len(train_loader)} batches per epoch)")
        
        # Set up mixed precision training if configured and GPU is available
        use_mixed_precision = CONFIG.get('use_mixed_precision', False) and torch.cuda.is_available()
        if use_mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Using mixed precision training for better performance")
        
        # Get gradient accumulation steps
        grad_accum_steps = CONFIG.get('gradient_accumulation_steps', 1)
        effective_batch_size = batch_size * grad_accum_steps
        logger.info(f"Using gradient accumulation with {grad_accum_steps} steps (effective batch size: {effective_batch_size})")
        
        # Training loop
        for epoch in range(CONFIG['epochs']):
            # Training phase
            model.train()
            train_loss = 0
            running_loss = 0
            n_samples = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
            
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_pbar):
                # Get inputs and labels
                embeddings_batch = batch['embedding'].to(device, non_blocking=True)
                category_labels = batch['label'].to(device, non_blocking=True)
                
                # Enable mixed precision if configured
                if use_mixed_precision:
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        outputs = model(embeddings_batch)
                        
                        # Calculate loss
                        loss = category_criterion(outputs, category_labels)
                        loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Only step optimizer and scaler after accumulating gradients
                    if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                        # Clip gradients to prevent exploding gradients
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                        
                        # Update weights with gradient scaling
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        # Update learning rate
                        scheduler.step()
                else:
                    # Regular forward pass
                    outputs = model(embeddings_batch)
                    
                    # Calculate loss
                    loss = category_criterion(outputs, category_labels)
                    loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
                    
                    # Backward pass
                    loss.backward()
                    
                    # Only step optimizer after accumulating gradients
                    if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                        
                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Update learning rate
                        scheduler.step()
                
                # Track statistics
                running_loss += loss.item() * grad_accum_steps  # Un-scale the loss
                n_samples += embeddings_batch.size(0)
                
                # Update progress bar with running statistics
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    train_pbar.set_postfix({"loss": running_loss / n_samples})
                
                train_loss += loss.item() * grad_accum_steps  # Un-scale the loss
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]")
            
            with torch.no_grad():
                for batch in val_pbar:
                    # Get inputs and labels
                    embeddings_batch = batch['embedding'].to(device, non_blocking=True)
                    category_labels = batch['label'].to(device, non_blocking=True)
                    
                    # Enable mixed precision for inference if configured
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            # Forward pass
                            outputs = model(embeddings_batch)
                            
                            # Calculate loss
                            loss = category_criterion(outputs, category_labels)
                    else:
                        # Regular forward pass
                        outputs = model(embeddings_batch)
                        
                        # Calculate loss
                        loss = category_criterion(outputs, category_labels)
                    
                    val_loss += loss.item()
                    
                    # Get predictions
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(category_labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Update training history
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['f1_score'].append(f1)
            self.training_history['iterations'].append(self.iterations)
            
            logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_accuracy = accuracy # Update best accuracy when loss improves
                # Only save model state dict to save memory
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model state if early stopping occurred
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Store the trained model
        self.models[model_type] = model
        self.current_model = model
        
        # Save the final model state
        model_path = os.path.join(self.models_dir, f'{model_type}_model.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Trained {model_type} model saved to {model_path}")
        
        # Increment iteration counter after successful training
        self.iterations += 1
        
        # Cleanup dataset resources
        if hasattr(train_dataset, 'cleanup'):
            train_dataset.cleanup()
        if hasattr(val_dataset, 'cleanup'):
            val_dataset.cleanup()
            
        # Empty CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_accuracy
    
    def load_model(self, model_type='transformer'):
        """Load a trained model"""
        model_path = os.path.join(self.models_dir, f"{model_type}_model_final.pth")
        metadata_path = os.path.join(self.models_dir, f"{model_type}_model_metadata.json")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            logger.warning(f"Model files not found for {model_type}. Train the model first.")
            return False
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Initialize model
            input_size = metadata['input_size']
            hidden_size = metadata['hidden_size']
            num_categories = metadata['num_categories']
            num_concepts = metadata.get('num_concepts', 1000)
            
            if model_type == 'transformer':
                model = TransformerClassifier(input_size, hidden_size, num_categories)
            elif model_type == 'multitask':
                model = MultiTaskModel(input_size, hidden_size, num_categories, num_concepts)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load state dict
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            model.eval()
            
            # Store model info
            self.models[model_type] = {
                'model': model,
                'metadata': metadata
            }
            
            # Set categories and mapping
            self.categories = metadata['categories']
            self.cat_to_idx = metadata['cat_to_idx']
            
            logger.info(f"Loaded {model_type} model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_category(self, embedding, model_type='transformer'):
        """Predict category for a paper embedding"""
        if model_type not in self.models:
            if not self.load_model(model_type):
                logger.error(f"No {model_type} model available for prediction")
                return None, None
        
        model = self.models[model_type]['model']
        model.eval()
        
        with torch.no_grad():
            # Convert to tensor and move to device
            embedding_tensor = torch.tensor(embedding, dtype=torch.float).unsqueeze(0).to(device)
            
            # Get predictions
            if model_type == 'transformer':
                outputs = model(embedding_tensor)
            else:  # multitask
                outputs = model(embedding_tensor)['category_logits']
            
            # Get probabilities and predicted class
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            pred_idx = np.argmax(probs)
            pred_category = self.categories[pred_idx]
            
            return pred_category, probs
    
    def evaluate(self, model_type='enhanced'):
        """Evaluate model with proper error handling"""
        if not hasattr(self, 'model') or self.model is None:
            logger.warning(f"Model files not found for {model_type}. Train the model first.")
            return None
            
        try:
            # Evaluation logic...
            return {
                'overall_f1': f1,
                'accuracy': accuracy,
                'per_category': category_metrics
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def generate_learning_curves(self):
        """Generate learning curves from training history"""
        import matplotlib.pyplot as plt
        
        if not self.training_history or not self.training_history['iterations']:
            logger.warning("No training history available to generate learning curves.")
            return
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axs[0].plot(self.training_history['iterations'], self.training_history['loss'])
        axs[0].set_title('Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].grid(True)
        
        # Plot accuracy and F1
        axs[1].plot(self.training_history['iterations'], self.training_history['accuracy'], label='Accuracy')
        axs[1].plot(self.training_history['iterations'], self.training_history['f1'], label='F1 Score')
        axs[1].set_title('Validation Metrics')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Score')
        axs[1].legend()
        axs[1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'learning_curves.png'))
        logger.info(f"Learning curves saved to {os.path.join(self.models_dir, 'learning_curves.png')}")
        plt.close()

    def incremental_train(self, new_papers, model_type='enhanced', learning_rate=5e-5):
        """Train model incrementally with new papers while preserving knowledge
        
        Args:
            new_papers (list): List of new paper IDs
            model_type (str): Model type to train
            learning_rate (float): Learning rate for incremental training
        
        Returns:
            bool: True if training was successful
        """
        logger.info(f"Starting incremental training with {len(new_papers)} new papers")
        
        # Prepare data from new papers only
        new_embeddings = []
        new_labels = []
        
        # Create category mapping if needed
        if not hasattr(self, 'categories') or not self.categories:
            logger.info("No existing categories found. Initializing from KB.")
            self.categories = sorted(list(self.kb.category_index.keys()))
            if not self.categories:
                logger.error("No categories available for incremental training")
                return False
        
        category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        
        # Extract data from new papers
        for paper_id in new_papers:
            # Get embedding
            embedding = self.kb.get_embedding(paper_id)
            if embedding is None:
                continue
                
            # Get paper metadata
            paper_meta = self.kb.papers.get(paper_id)
            if not paper_meta or not paper_meta.get('categories'):
                continue
                
            # Get valid categories
            paper_categories = paper_meta.get('categories', [])
            valid_categories = [cat for cat in paper_categories if cat in self.categories]
            if not valid_categories:
                continue
                
            # Use first valid category
            category_idx = category_to_idx[valid_categories[0]]
            
            # Add to new training data
            new_embeddings.append(embedding)
            new_labels.append(category_idx)
        
        if not new_embeddings:
            logger.warning("No valid data extracted from new papers")
            return False
            
        # Convert to numpy arrays
        X_new = np.array(new_embeddings)
        y_new = np.array(new_labels)
        
        logger.info(f"Extracted {len(X_new)} valid samples from new papers")
        
        # Load existing model
        if not self.load_model(model_type):
            logger.warning(f"No existing {model_type} model found, training from scratch")
            return self.train(model_type)
        
        # Initialize model parameters
        input_size = X_new.shape[1]
        hidden_size = CONFIG.get('hidden_size', 768)
        num_categories = len(self.categories)
        
        # Create datasets
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create new data
        new_dataset = PaperDataset(X_new, y_new)
        new_loader = DataLoader(
            new_dataset,
            batch_size=CONFIG.get('batch_size', 32),
            shuffle=True,
            num_workers=CONFIG.get('num_workers', 4)
        )
        
        # Create replay buffer from old data (if available)
        replay_loader = None
        if hasattr(self, 'X') and hasattr(self, 'y') and len(self.X) > 0:
            # Sample a subset of old data for replay buffer
            replay_size = min(CONFIG.get('replay_buffer_size', 1000), len(self.X))
            indices = np.random.choice(len(self.X), replay_size, replace=False)
            X_replay = self.X[indices]
            y_replay = self.y[indices]
            
            replay_dataset = PaperDataset(X_replay, y_replay)
            replay_loader = DataLoader(
                replay_dataset,
                batch_size=CONFIG.get('batch_size', 32),
                shuffle=True,
                num_workers=CONFIG.get('num_workers', 4)
            )
            
            logger.info(f"Created replay buffer with {replay_size} samples")
        
        # Ensure model is in training mode
        self.model.train()
        
        # Set up optimizer with lower learning rate
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=CONFIG.get('weight_decay', 1e-5),
        )
        
        # Use fewer epochs for incremental training
        epochs = CONFIG.get('incremental_epochs', 3)
        
        # Initialize loss function
        criterion = nn.CrossEntropyLoss()
        
        # Initialize training history for this session
        history = {
            'epochs': [],
            'new_loss': [],
            'replay_loss': [],
            'val_loss': []
        }
        
        # Train for specified epochs
        device = self.device
        
        # Enable mixed precision if available
        use_mixed_precision = CONFIG.get('use_mixed_precision', False) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Train the model
        for epoch in range(epochs):
            # Train on new data
            self.model.train()
            total_new_loss = 0
            total_replay_loss = 0
            
            # Process new data
            for batch in new_loader:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(embeddings)
                        loss = criterion(outputs, labels)
                    
                    # Scale gradients and optimize
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward pass
                    outputs = self.model(embeddings)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                total_new_loss += loss.item()
            
            # Process replay buffer (if available)
            if replay_loader:
                for batch in replay_loader:
                    embeddings = batch['embedding'].to(device)
                    labels = batch['label'].to(device)
                    
                    optimizer.zero_grad()
                    
                    # Mixed precision forward pass
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(embeddings)
                            loss = criterion(outputs, labels)
                        
                        # Scale gradients and optimize
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard forward pass
                        outputs = self.model(embeddings)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    
                    total_replay_loss += loss.item()
            
            # Calculate average losses
            avg_new_loss = total_new_loss / len(new_loader)
            avg_replay_loss = total_replay_loss / len(replay_loader) if replay_loader else 0
            
            # Update history
            history['epochs'].append(epoch + 1)
            history['new_loss'].append(avg_new_loss)
            history['replay_loss'].append(avg_replay_loss)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}, New Loss: {avg_new_loss:.4f}, Replay Loss: {avg_replay_loss:.4f}")
        
        # Update combined dataset for future training
        if hasattr(self, 'X') and hasattr(self, 'y'):
            # Combine new and existing data
            self.X = np.vstack([self.X, X_new])
            self.y = np.concatenate([self.y, y_new])
            logger.info(f"Updated training data: {len(self.X)} samples total")
        else:
            # Initialize with new data
            self.X = X_new
            self.y = y_new
            logger.info(f"Initialized training data with {len(self.X)} samples")
        
        # Save the incrementally trained model
        self.save_model(model_type)
        
        # Update training history
        if hasattr(self, 'training_history'):
            # Append to existing history
            for key in history:
                if key in self.training_history:
                    self.training_history[key].extend(history[key])
                else:
                    self.training_history[key] = history[key]
        else:
            # Initialize with current history
            self.training_history = history
        
        return True
