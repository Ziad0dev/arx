import torch
import faiss
import numpy as np
import os
import json
import pickle
import logging
import math
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
from advanced_ai_analyzer import CONFIG
from utils.db_connector import DatabaseConnector
from utils.embedding_manager import EmbeddingManager

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Enhanced knowledge base using FAISS for efficient similarity search and MongoDB for persistence"""
    
    def __init__(self, db_file=None, faiss_index_file=None):
        """Initialize the Knowledge Base.
        
        Args:
            db_file (str, optional): Path to the database file. Defaults to CONFIG['knowledge_base_file'].
            faiss_index_file (str, optional): Path to the FAISS index file. Defaults to CONFIG['faiss_index_file'].
        """
        self.db_file = db_file if db_file is not None else CONFIG['knowledge_base_file']
        self.faiss_index_file = faiss_index_file if faiss_index_file is not None else CONFIG['faiss_index_file']
        self.papers = {}  # Stores metadata {paper_id: paper_data} - memory cache
        self.faiss_index = None
        self.paper_ids_in_index = [] # Maps FAISS index position to paper_id
        self.embedding_dim = CONFIG.get('embedding_size', 768) # Ensure consistent embedding size

        # Non-vector indexes remain the same
        self.concept_index = defaultdict(list)
        self.category_index = defaultdict(list)
        self.author_index = defaultdict(list)
        self.timestamp_index = []
        
        # For concept relationships
        self.concept_relations = defaultdict(list)
        
        # Initialize embedding manager for encoding queries
        self.embedding_manager = EmbeddingManager()
        
        # Initialize database connector if enabled
        self.db_connector = None
        if CONFIG.get('use_database', True):
            self.db_connector = DatabaseConnector()
            self.db_connector.connect()
            # Load indexes from database
            if self.db_connector.connected:
                self._load_from_database()
            else:
                # Fall back to file-based loading
                self.load()
        else:
            # Use file-based loading only
            self.load()
    
    def _load_from_database(self):
        """Load papers and indexes from MongoDB database"""
        if not self.db_connector or not self.db_connector.connected:
            logger.warning("Database connection not available, skipping database load")
            return False
        
        try:
            # Get all papers from database with lightweight projection (no embeddings)
            papers = self.db_connector.get_all_papers()
            logger.info(f"Loading {len(papers)} papers from database")
            
            # Add to in-memory indexes
            for paper in papers:
                paper_id = paper['id']
                self.papers[paper_id] = paper
                self._update_indexes(paper_id, paper, paper.get('concepts', []))
            
            # Initialize FAISS index
            self._initialize_faiss_index()
            
            # Load paper IDs in FAISS index
            logger.info("Initializing FAISS with paper IDs from database")
            embeddings_docs = self.db_connector.db.embeddings.find({}, {"paper_id": 1})
            self.paper_ids_in_index = [doc['paper_id'] for doc in embeddings_docs]
            
            # Load concept relationships if available
            self._load_concept_relationships()
            
            logger.info(f"Successfully loaded knowledge base from database: {len(self.papers)} papers, {len(self.concept_index)} concepts")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            return False
    
    def _load_concept_relationships(self):
        """Load concept relationships from database"""
        if not self.db_connector or not self.db_connector.connected:
            return False
        
        try:
            relationships = self.db_connector.get_concept_relationships()
            logger.info(f"Loading {len(relationships)} concept relationships from database")
            
            for rel in relationships:
                source = rel['source']
                target = rel['target']
                relation_type = rel['relation']
                confidence = rel['confidence']
                papers = rel.get('papers', [])
                
                self.concept_relations[source].append({
                    'source': source,
                    'target': target,
                    'relation': relation_type,
                    'confidence': confidence,
                    'papers': papers
                })
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load concept relationships: {e}")
            return False

    def _initialize_faiss_index(self):
        """Initialize an empty FAISS index if one doesn't exist."""
        if self.faiss_index is None:
            # Use more efficient index type based on dimensionality
            # For higher dimensions, we use IVF for faster search with minimal accuracy loss
            if self.embedding_dim > 100:
                # Use GPU if available
                if CONFIG.get('use_gpu', False) and torch.cuda.is_available():
                    # First create CPU index
                    quantizer = faiss.IndexFlatL2(self.embedding_dim)
                    nlist = max(4, min(1000, int(len(self.papers) / 10))) if self.papers else 100
                    index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
                    
                    # Convert to GPU
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info(f"Initialized GPU FAISS IVFFlat index with dimension {self.embedding_dim}, nlist={nlist}")
                else:
                    # CPU implementation
                    quantizer = faiss.IndexFlatL2(self.embedding_dim)
                    nlist = max(4, min(1000, int(len(self.papers) / 10))) if self.papers else 100
                    self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
                    logger.info(f"Initialized CPU FAISS IVFFlat index with dimension {self.embedding_dim}, nlist={nlist}")
            else:
                # For lower dimensions, simple flat index is efficient enough
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info(f"Initialized FAISS flat index with dimension {self.embedding_dim}")
            
            self.paper_ids_in_index = []
    
    def _normalize_embedding(self, embedding):
        """Normalize embedding to unit length for cosine similarity search via L2."""
        if embedding is None:
            return None
        
        # Convert to numpy array if it's not already
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
            
        emb_array = np.array(embedding, dtype=np.float32)
        
        # Handle non-vector cases
        if len(emb_array.shape) == 0:
            emb_array = np.array([emb_array], dtype=np.float32)
        elif len(emb_array.shape) > 1:
            # If it's a batch of vectors or a 2D matrix, take the first one or average
            emb_array = np.mean(emb_array, axis=0)
        
        # Reshape to ensure proper dimensionality
        if emb_array.shape[0] != self.embedding_dim:
            # If dimensions don't match, try to reshape or pad/truncate
            if len(emb_array) > self.embedding_dim:
                emb_array = emb_array[:self.embedding_dim]
            else:
                # Pad with zeros
                padded = np.zeros(self.embedding_dim, dtype=np.float32)
                padded[:len(emb_array)] = emb_array
                emb_array = padded
            
            logger.warning(f"Embedding dimension mismatch. Reshaped from {len(embedding)} to {self.embedding_dim}.")
        
        # Normalize to unit length
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            return emb_array / norm
        return emb_array  # Return as is if norm is 0

    def add_papers(self, processed_papers):
        """Add multiple papers to the knowledge base and FAISS index."""
        if not processed_papers:
            return 0
            
        added_count = 0
        embeddings_to_add = []
        ids_to_add = []

        for paper in processed_papers:
            if not paper or 'id' not in paper:
                continue

            paper_id = paper['id']
            if paper_id in self.papers:
                continue # Skip duplicates

            # Store paper metadata (excluding the large embedding)
            paper_data = {
                'id': paper_id,
                'title': paper.get('title', 'Untitled'),
                'concepts': [c for c in paper.get('concepts', []) if c],
                'categories': list(set(paper.get('metadata', {}).get('categories', []))),
                'published': paper.get('metadata', {}).get('published', ''),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('metadata', {}).get('authors', [])
            }
            
            # Store in memory cache
            self.papers[paper_id] = paper_data

            # Update non-vector indexes
            self._update_indexes(paper_id, paper.get('metadata', {}), paper.get('concepts', []))

            # Prepare embedding for FAISS
            embedding = paper.get('embedding')
            if embedding is not None:
                normalized_embedding = self._normalize_embedding(embedding)
                if normalized_embedding is not None and len(normalized_embedding) == self.embedding_dim:
                    embeddings_to_add.append(normalized_embedding)
                    ids_to_add.append(paper_id)
                else:
                    logger.warning(f"Skipping embedding for paper {paper_id}: incorrect dimension after normalization.")

            # Store in database if enabled
            if self.db_connector and self.db_connector.connected:
                # Make a copy with embedding for database storage
                db_paper = paper_data.copy()
                if embedding is not None:
                    db_paper['embedding'] = embedding
                    
                self.db_connector.store_paper(db_paper)

            added_count += 1

        # Add embeddings to FAISS index in batch
        if embeddings_to_add:
            if self.faiss_index is None:
                self._initialize_faiss_index()
                
            # Training for IVF index if needed
            if isinstance(self.faiss_index, faiss.IndexIVFFlat) and not self.faiss_index.is_trained:
                if len(embeddings_to_add) > 10:  # Need enough vectors to train
                    embeddings_np = np.array(embeddings_to_add).astype('float32')
                    try:
                        logger.info(f"Training FAISS IVF index with {len(embeddings_to_add)} vectors")
                        self.faiss_index.train(embeddings_np)
                    except Exception as e:
                        logger.error(f"Failed to train FAISS index: {e}")
            
            # Add embeddings to index
            try:
                embeddings_np = np.array(embeddings_to_add).astype('float32')
                self.faiss_index.add(embeddings_np)
                self.paper_ids_in_index.extend(ids_to_add)
                logger.debug(f"Added {len(embeddings_to_add)} embeddings to FAISS index. Total size: {self.faiss_index.ntotal}")
            except Exception as e:
                logger.error(f"Failed to add embeddings to FAISS index: {e}")

        # Save periodically if not using database
        if added_count > 0 and (not self.db_connector or not self.db_connector.connected):
            self.save()

        return added_count

    def get_embedding(self, paper_id):
        """Get embedding for a paper, either from database or local cache"""
        if not paper_id:
            return None
            
        # Try database first if available
        if self.db_connector and self.db_connector.connected:
            embedding = self.db_connector.get_paper_embedding(paper_id)
            if embedding is not None:
                return embedding
        
        # Fall back to file-based lookup - not efficient, but keeping for backward compatibility
        embeddings_file = CONFIG['embeddings_file']
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    all_embeddings = pickle.load(f)
                    return all_embeddings.get(paper_id)
            except Exception as e:
                logger.error(f"Error reading embeddings file {embeddings_file}: {e}")
                
        return None

    def get_papers_by_concept(self, concept, limit=100):
        """Get papers related to a specific concept"""
        if not concept:
            return []
            
        paper_ids = self.concept_index.get(concept, [])[:limit]
        return [self.papers[pid] for pid in paper_ids if pid in self.papers]

    def get_papers_by_category(self, category, limit=100):
        """Get papers in a specific category"""
        if not category:
            return []
            
        paper_ids = self.category_index.get(category, [])[:limit]
        return [self.papers[pid] for pid in paper_ids if pid in self.papers]

    def get_papers_by_author(self, author, limit=100):
        """Get papers by a specific author"""
        if not author:
            return []
            
        paper_ids = self.author_index.get(author, [])[:limit]
        return [self.papers[pid] for pid in paper_ids if pid in self.papers]

    def get_recent_papers(self, limit=100):
        """Get most recent papers by publication date"""
        return [self.papers[pid] for pid, _ in self.timestamp_index[:limit] if pid in self.papers]

    def get_concepts_by_category(self, category, limit=100):
        """Get concepts associated with a specific category"""
        papers = self.get_papers_by_category(category)
        concept_counter = Counter()
        for paper in papers:
            for concept in paper.get('concepts', []):
                concept_counter[concept] += 1
        return [c for c, _ in concept_counter.most_common(limit)]

    def get_top_concepts(self, category=None, n=10):
        """Get most frequent concepts, optionally filtered by category"""
        if category:
            papers = self.get_papers_by_category(category)
        else:
            papers = self.papers.values()
            
        concept_counter = Counter()
        for paper in papers:
            for concept in paper.get('concepts', []):
                concept_counter[concept] += 1
                
        return [c for c, _ in concept_counter.most_common(n)]

    def get_similar_papers(self, paper_id, top_n=10):
        """Find semantically similar papers using FAISS"""
        if not paper_id or paper_id not in self.papers:
            return []
            
        query_embedding = self.get_embedding(paper_id)
        if query_embedding is None:
            logger.warning(f"No embedding found for paper {paper_id}")
            return []
            
        query_embedding = self._normalize_embedding(query_embedding)
        if query_embedding is None:
            return []
            
        return self.semantic_search(query_embedding, top_n)

    def semantic_search(self, query_embedding, top_n=10):
        """Search for papers similar to a query embedding"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            logger.warning("FAISS index not initialized or empty")
            return []
            
        if len(self.paper_ids_in_index) == 0:
            logger.warning("No paper IDs in FAISS index")
            return []
            
        if query_embedding is None:
            logger.warning("Query embedding is None")
            return []
            
        # Normalize and reshape query
        query_embedding = self._normalize_embedding(query_embedding)
        query_embedding = np.array([query_embedding], dtype=np.float32)
            
        # Perform search
        try:
            # Set nprobe higher for IVF indexes to get better accuracy at slight perf cost
            if hasattr(self.faiss_index, 'nprobe'):
                self.faiss_index.nprobe = min(20, max(10, int(self.faiss_index.ntotal / 50)))
                
            distances, indices = self.faiss_index.search(query_embedding, min(top_n, self.faiss_index.ntotal))
            
            # Convert results to paper data
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.paper_ids_in_index):
                    continue # Skip invalid indices
                    
                paper_id = self.paper_ids_in_index[idx]
                if paper_id in self.papers:
                    paper = self.papers[paper_id].copy()
                    paper['similarity'] = float(1.0 - distances[0][i])  # Convert distance to similarity
                    results.append(paper)
            
            return results
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []

    def search_by_text(self, query_text, top_n=10):
        """Search for papers by converting text to embedding"""
        if not query_text:
            return []
            
        # Generate embedding for query text
        query_embedding = self.embedding_manager.get_embedding_for_text(query_text)
        if query_embedding is None:
            logger.warning(f"Failed to generate embedding for query: {query_text}")
            return []
            
        # Perform semantic search with embedding
        return self.semantic_search(query_embedding, top_n)

    def add_concept_relationship(self, concept1, concept2, relation_type, confidence=1.0, paper_ids=None):
        """Add a relationship between two concepts"""
        if not concept1 or not concept2:
            return False
            
        relationship = {
            'source': concept1,
            'target': concept2,
            'relation': relation_type,
            'confidence': confidence,
            'papers': paper_ids or []
        }
        
        # Add to in-memory store
        self.concept_relations[concept1].append(relationship)
        
        # Add to database if available
        if self.db_connector and self.db_connector.connected:
            self.db_connector.store_concept_relationship(
                concept1, concept2, relation_type, confidence, paper_ids
            )
        
        return True

    def get_concept_relationships(self, concept=None, min_confidence=0.5):
        """Get relationships for a concept or all relationships"""
        if concept:
            # Filter by source concept
            return [rel for rel in self.concept_relations.get(concept, []) 
                    if rel['confidence'] >= min_confidence]
        else:
            # Get all relationships
            all_rels = []
            for concept, rels in self.concept_relations.items():
                for rel in rels:
                    if rel['confidence'] >= min_confidence:
                        all_rels.append(rel)
            return all_rels

    def extract_concept_relationships(self):
        """Extract relationships between concepts based on co-occurrence"""
        # Count concept co-occurrences in papers
        co_occurrences = defaultdict(int)
        concept_counts = defaultdict(int)
        
        # Iterate through papers to count concept occurrences and co-occurrences
        for paper_id, paper in self.papers.items():
            concepts = paper.get('concepts', [])
            for concept in concepts:
                concept_counts[concept] += 1
                
            # Count co-occurrences (pairs of concepts in same paper)
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    if concept1 != concept2:
                        pair = tuple(sorted([concept1, concept2]))
                        co_occurrences[pair] += 1
        
        # Calculate PMI (Pointwise Mutual Information) for pairs
        total_papers = len(self.papers)
        if total_papers == 0:
            logger.warning("No papers to extract concept relationships from")
            return False
            
        relationship_count = 0
        for (concept1, concept2), co_count in tqdm(co_occurrences.items(), desc="Extracting relationships"):
            # Only consider pairs with meaningful co-occurrence
            if co_count < 3:
                continue
                
            count1 = concept_counts[concept1]
            count2 = concept_counts[concept2]
            
            # Calculate PMI score
            pmi = math.log((co_count * total_papers) / (count1 * count2))
            
            # Add relationship if PMI is positive (concepts are positively correlated)
            if pmi > 0:
                confidence = min(1.0, pmi / 5)  # Normalize PMI to 0-1 range
                self.add_concept_relationship(concept1, concept2, 'related_to', confidence)
                self.add_concept_relationship(concept2, concept1, 'related_to', confidence)
                relationship_count += 2
        
        logger.info(f"Extracted {relationship_count} concept relationships")
        return True

    def save(self):
        """Save knowledge base to file."""
        # No need to save if using database and it's connected
        if self.db_connector and self.db_connector.connected:
            return True
            
        try:
            # Save knowledge base data
            kb_data = {
                'papers': self.papers,
                'concept_index': {k: list(v) for k, v in self.concept_index.items()},
                'category_index': {k: list(v) for k, v in self.category_index.items()},
                'author_index': {k: list(v) for k, v in self.author_index.items()},
                'timestamp_index': self.timestamp_index,
                'concept_relations': {k: list(v) for k, v in self.concept_relations.items()},
                'embedding_dim': self.embedding_dim
            }
            
            # Save knowledge base
            with open(self.db_file, 'w') as f:
                json.dump(kb_data, f)
                
            # Save FAISS index if it exists
            if self.faiss_index is not None and hasattr(self.faiss_index, 'ntotal') and self.faiss_index.ntotal > 0:
                # Save FAISS index (CPU version only)
                index_to_save = self.faiss_index
                
                # Convert GPU index to CPU if needed
                if hasattr(self.faiss_index, 'getDevice') and self.faiss_index.getDevice() >= 0:
                    index_to_save = faiss.index_gpu_to_cpu(self.faiss_index)
                    
                faiss.write_index(index_to_save, self.faiss_index_file)
                
                # Save paper IDs mapping
                with open(self.faiss_index_file + '.ids', 'wb') as f:
                    pickle.dump(self.paper_ids_in_index, f)
                    
            logger.info(f"Knowledge base saved: {len(self.papers)} papers, {len(self.concept_index)} concepts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
            return False

    def load(self):
        """Load knowledge base from file."""
        # Skip file loading if using database
        if self.db_connector and self.db_connector.connected:
            return self._load_from_database()
            
        try:
            # Load knowledge base data if file exists
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r') as f:
                    kb_data = json.load(f)
                    
                self.papers = kb_data.get('papers', {})
                self.concept_index = defaultdict(list, kb_data.get('concept_index', {}))
                self.category_index = defaultdict(list, kb_data.get('category_index', {}))
                self.author_index = defaultdict(list, kb_data.get('author_index', {}))
                self.timestamp_index = kb_data.get('timestamp_index', [])
                self.concept_relations = defaultdict(list, kb_data.get('concept_relations', {}))
                
                # Update embedding dimension from saved data, but don't override CONFIG
                saved_dim = kb_data.get('embedding_dim')
                if saved_dim and saved_dim != self.embedding_dim:
                    logger.warning(f"Saved embedding dimension ({saved_dim}) differs from CONFIG ({self.embedding_dim}). Using CONFIG value.")
            
            # Load FAISS index if file exists
            if os.path.exists(self.faiss_index_file):
                # Initialize FAISS index
                self._initialize_faiss_index()
                
                try:
                    # Load index
                    loaded_index = faiss.read_index(self.faiss_index_file)
                    
                    # Replace the current index with the loaded one
                    self.faiss_index = loaded_index
                    
                    # Load paper IDs mapping if exists
                    if os.path.exists(self.faiss_index_file + '.ids'):
                        with open(self.faiss_index_file + '.ids', 'rb') as f:
                            self.paper_ids_in_index = pickle.load(f)
                    else:
                        # Initialize with sequential indices
                        self.paper_ids_in_index = list(self.papers.keys())[:self.faiss_index.ntotal]
                        
                    logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
                    
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
                    self._initialize_faiss_index()
            
            logger.info(f"Knowledge base loaded from files: {len(self.papers)} papers, {len(self.concept_index)} concepts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            # Initialize empty structures
            self.papers = {}
            self.concept_index = defaultdict(list)
            self.category_index = defaultdict(list)
            self.author_index = defaultdict(list)
            self.timestamp_index = []
            self.concept_relations = defaultdict(list)
            self._initialize_faiss_index()
            return False

    def _update_indexes(self, paper_id, metadata, concepts):
        """Update non-vector indexes with new paper."""
        # Update concept index
        for concept in concepts:
            self.concept_index[concept].append(paper_id)

        # Update category index
        categories = metadata.get('categories', [])
        if isinstance(categories, str): # Handle legacy single category string
            categories = [categories]
        for category in categories:
            self.category_index[category].append(paper_id)

        # Update author index
        authors = metadata.get('authors', [])
        for author in authors:
            self.author_index[author].append(paper_id)

        # Update timestamp index using 'published' date if available
        published_date = metadata.get('published')
        if published_date:
            # Attempt to parse the date for proper sorting, fallback to string comparison
            try:
                # Assuming ISO format like '2023-10-27T10:00:00Z' or similar
                ts = datetime.fromisoformat(published_date.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError):
                ts = published_date # Fallback to string sorting if parsing fails
            self.timestamp_index.append((paper_id, ts))
            self.timestamp_index.sort(key=lambda x: x[1], reverse=True)

    def count_concepts(self):
        """Count the number of concepts in the knowledge base"""
        return len(self.concept_index)

    def concept_novelty_score(self, concepts):
        """Calculate novelty score of concepts based on rarity"""
        if not concepts:
            return 0.0
            
        total_papers = max(1, len(self.papers))
        scores = []
        
        for concept in concepts:
            # Get papers with this concept
            papers_with_concept = len(self.concept_index.get(concept, []))
            
            # Calculate rarity score
            if papers_with_concept == 0:
                # Completely novel concept
                score = 1.0
            else:
                # Lower score for common concepts
                score = 1.0 - (papers_with_concept / total_papers)
            
            scores.append(score)
        
        # Return average novelty score
        return sum(scores) / len(scores) if scores else 0.0
