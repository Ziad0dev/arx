import torch
from advanced_ai_analyzer import *

class KnowledgeBase:
    """Enhanced knowledge base with advanced storage and retrieval capabilities"""
    
    def __init__(self, db_file=CONFIG['knowledge_base_file'], embeddings_file=CONFIG['embeddings_file']):
        self.db_file = db_file
        self.embeddings_file = embeddings_file
        self.papers = {}
        self.paper_embeddings = {}
        self.concept_index = defaultdict(list)
        self.category_index = defaultdict(list)
        self.author_index = defaultdict(list)
        self.timestamp_index = []  # List of (timestamp, paper_id) tuples
        self.load()
        
        # Initialize vector store for semantic search
        self.vector_store_initialized = False
        if len(self.paper_embeddings) > 0:
            self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize vector store for semantic search"""
        try:
            paper_ids = list(self.paper_embeddings.keys())
            if not paper_ids:
                logger.warning("No paper embeddings available for vector store")
                self.vector_store_initialized = False
                return
            
            # Check if all embeddings have the same dimension
            embedding_sizes = [len(self.paper_embeddings[pid]) for pid in paper_ids]
            if len(set(embedding_sizes)) > 1:
                logger.warning("Embeddings have inconsistent dimensions. Fixing...")
                self._fix_embedding_dimensions()
                paper_ids = list(self.paper_embeddings.keys())
                
            # Now try again with fixed embeddings
            try:
                embeddings = np.array([self.paper_embeddings[pid] for pid in paper_ids])
                
                # Normalize embeddings for cosine similarity
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Avoid division by zero
                norms[norms == 0] = 1.0
                self.normalized_embeddings = embeddings / norms
                self.vector_store_paper_ids = paper_ids
                self.vector_store_initialized = True
                logger.info(f"Vector store initialized with {len(paper_ids)} papers")
            except Exception as e:
                logger.error(f"Failed to create embedding array: {e}")
                self.vector_store_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store_initialized = False
            
    def _fix_embedding_dimensions(self):
        """Fix inconsistent embedding dimensions"""
        if not self.paper_embeddings:
            return
            
        # Use CONFIG['hidden_size'] as the target dimension
        target_dim = CONFIG['hidden_size']
        
        # Fix each embedding
        fixed_embeddings = {}
        for paper_id, embedding in self.paper_embeddings.items():
            if isinstance(embedding, list) and len(embedding) != target_dim:
                # Create a new embedding of the correct size
                new_embedding = np.zeros(target_dim)
                # Copy as much as we can from the original
                copy_len = min(len(embedding), target_dim)
                new_embedding[:copy_len] = embedding[:copy_len]
                fixed_embeddings[paper_id] = new_embedding.tolist()
            else:
                # Keep the original if it's already the right size
                fixed_embeddings[paper_id] = embedding
        
        # Replace the embeddings
        self.paper_embeddings = fixed_embeddings
        logger.info(f"Fixed dimensions for {len(fixed_embeddings)} embeddings")
    
    def add_papers(self, processed_papers):
        """Add multiple papers to the knowledge base
        
        Args:
            processed_papers: List of processed paper dictionaries
            
        Returns:
            Number of papers added
        """
        added_count = 0
        for paper in processed_papers:
            if not paper:
                continue
                
            # Extract paper components
            paper_id = paper.get('id')
            metadata = paper.get('metadata', {})
            concepts = paper.get('concepts', [])
            embedding = paper.get('embedding')
            
            # Add the paper
            if self.add_paper(paper):
                added_count += 1
        
        # Force save if we added papers
        if added_count > 0:
            self.save()
            
            # Reinitialize vector store if needed
            if embedding is not None and not self.vector_store_initialized:
                self._initialize_vector_store()
                
        return added_count
        
    def add_paper(self, paper):
        """Add a single paper to the knowledge base"""
        if not paper or 'id' not in paper:
            return False
            
        paper_id = paper['id']
        if paper_id in self.papers:
            return False
            
        self.papers[paper_id] = {
            'id': paper_id,
            'title': paper.get('title', 'Untitled'),  
            'concepts': [c[:20] for c in paper.get('concepts', [])],  
            'categories': list(set(paper.get('metadata', {}).get('categories', []))),
            'published': paper.get('metadata', {}).get('published', ''),
            'embedding': torch.tensor(paper.get('embedding', None)).half() if paper.get('embedding', None) is not None else None  # Store as float16
        }
        
        # Update indexes
        self._update_indexes(paper_id, paper.get('metadata', {}), paper.get('concepts', []))
        
        # Save to disk periodically
        if len(self.papers) % 10 == 0:
            self.save()
            
        return True
    
    def _update_indexes(self, paper_id, metadata, concepts):
        """Update all indexes with new paper"""
        # Update concept index
        for concept in concepts:
            self.concept_index[concept].append(paper_id)
        
        # Update category index if categories exist
        if 'categories' in metadata:
            for category in metadata['categories']:
                self.category_index[category].append(paper_id)
        
        # Update author index if authors exist
        if 'authors' in metadata:
            for author in metadata['authors']:
                self.author_index[author].append(paper_id)
                
        # Update timestamp index
        self.timestamp_index.append((time.time(), paper_id))
        
        # Update timestamp index
        timestamp = time.time()
        self.timestamp_index.append((timestamp, paper_id))
        self.timestamp_index.sort(reverse=True)
    
    def get_papers_by_concept(self, concept, limit=100):
        """Get papers related to a specific concept"""
        return self.concept_index.get(concept, [])[:limit]
    
    def get_papers_by_category(self, category, limit=100):
        """Get papers in a specific category"""
        return self.category_index.get(category, [])[:limit]
    
    def get_papers_by_author(self, author, limit=100):
        """Get papers by a specific author"""
        return self.author_index.get(author, [])[:limit]
    
    def get_recent_papers(self, limit=100):
        """Get most recently added papers"""
        return [pid for _, pid in self.timestamp_index[:limit]]
    
    def get_concepts_by_category(self, category, limit=100):
        """Get concepts associated with a category"""
        concepts = []
        for paper_id in self.category_index.get(category, []):
            if paper_id in self.papers:
                concepts.extend(self.papers[paper_id]['concepts'])
        return list(set(concepts))[:limit]
    
    def get_top_concepts(self, category=None, n=10):
        """Get top concepts overall or by category"""
        if category:
            concepts = self.get_concepts_by_category(category)
        else:
            concepts = [concept for paper in self.papers.values() 
                       for concept in paper['concepts']]
        
        counter = Counter(concepts)
        return [word for word, _ in counter.most_common(n)]
    
    def get_similar_papers(self, paper_id, top_n=10):
        """Get papers similar to the given paper using vector similarity"""
        if not self.vector_store_initialized or paper_id not in self.paper_embeddings:
            return []
        
        # Get embedding for the query paper
        query_embedding = np.array(self.paper_embeddings[paper_id])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarities
        similarities = np.dot(self.normalized_embeddings, query_embedding)
        
        # Get top similar papers (excluding the query paper)
        similar_indices = np.argsort(similarities)[::-1]
        similar_papers = []
        
        for idx in similar_indices:
            pid = self.vector_store_paper_ids[idx]
            if pid != paper_id:
                similar_papers.append(pid)
                if len(similar_papers) >= top_n:
                    break
        
        return similar_papers
    
    def semantic_search(self, query_embedding, top_n=10):
        """Search for papers similar to the query embedding"""
        if not self.vector_store_initialized:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarities
        similarities = np.dot(self.normalized_embeddings, query_embedding)
        
        # Get top similar papers
        similar_indices = np.argsort(similarities)[::-1][:top_n]
        return [self.vector_store_paper_ids[idx] for idx in similar_indices]
    
    def get_knowledge_graph(self, limit=1000):
        """Generate a knowledge graph representation"""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Add paper nodes
        paper_count = 0
        for paper_id, paper_data in self.papers.items():
            if paper_count >= limit:
                break
                
            graph['nodes'].append({
                'id': paper_id,
                'type': 'paper',
                'label': paper_data['title'],
                'categories': paper_data['categories']
            })
            paper_count += 1
        
        # Add concept nodes for top concepts
        top_concepts = self.get_top_concepts(n=min(100, limit//10))
        for concept in top_concepts:
            graph['nodes'].append({
                'id': f"concept_{concept}",
                'type': 'concept',
                'label': concept
            })
        
        # Add edges between papers and concepts
        edge_count = 0
        for paper_id in [node['id'] for node in graph['nodes'] if node['type'] == 'paper']:
            if paper_id in self.papers:
                for concept in self.papers[paper_id]['concepts']:
                    if f"concept_{concept}" in [node['id'] for node in graph['nodes']]:
                        graph['edges'].append({
                            'source': paper_id,
                            'target': f"concept_{concept}",
                            'type': 'has_concept'
                        })
                        edge_count += 1
                        if edge_count >= limit * 2:
                            break
            if edge_count >= limit * 2:
                break
        
        return graph
    
    def generate_research_summary(self, category=None, limit=100):
        """Generate a research summary for a category or overall"""
        summary = {
            'total_papers': len(self.papers),
            'top_concepts': self.get_top_concepts(category, n=20),
            'top_authors': self._get_top_authors(category, n=10),
            'category_distribution': self._get_category_distribution(),
            'recent_papers': self._get_recent_paper_summaries(limit=10)
        }
        
        if category:
            summary['category'] = category
            summary['papers_in_category'] = len(self.category_index.get(category, []))
        
        return summary
    
    def _get_top_authors(self, category=None, n=10):
        """Get top authors overall or in a specific category"""
        if category:
            paper_ids = self.category_index.get(category, [])
            authors = [author for pid in paper_ids if pid in self.papers
                      for author in self.papers[pid]['metadata']['authors']]
        else:
            authors = [author for paper in self.papers.values()
                      for author in paper['metadata']['authors']]
        
        return [author for author, _ in Counter(authors).most_common(n)]
    
    def _get_category_distribution(self):
        """Get distribution of papers across categories"""
        categories = [cat for paper in self.papers.values()
                     for cat in paper['categories']]
        
        counter = Counter(categories)
        return {cat: count for cat, count in counter.most_common(20)}
    
    def _get_recent_paper_summaries(self, limit=10):
        """Get summaries of recently added papers"""
        recent_ids = self.get_recent_papers(limit=limit)
        summaries = []
        
        for pid in recent_ids:
            if pid in self.papers:
                paper = self.papers[pid]
                summaries.append({
                    'id': pid,
                    'title': paper['title'],
                    'authors': paper['metadata']['authors'],
                    'categories': paper['categories'],
                    'top_concepts': paper['concepts'][:5] if paper['concepts'] else []
                })
        
        return summaries
    
    def count_concepts(self):
        """Count all unique concepts in knowledge base"""
        return len(self.concept_index)
    
    def save(self):
        """Save knowledge base to disk"""
        try:
            # Save papers data
            with open(self.db_file, 'w') as f:
                json.dump(self.papers, f)
            
            # Save embeddings separately (they can be large)
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.paper_embeddings, f)
                
            logger.info(f"Saved knowledge base with {len(self.papers)} papers")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    def load(self):
        """Load knowledge base from disk"""
        # Load papers data
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.papers = json.load(f)
                logger.info(f"Loaded knowledge base with {len(self.papers)} papers")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {self.db_file}: {e}. Initializing empty knowledge base.")
                self.papers = {}
                os.rename(self.db_file, f"{self.db_file}.corrupted")
            except Exception as e:
                logger.error(f"Unexpected error loading {self.db_file}: {e}. Starting fresh.")
                self.papers = {}
        else:
            logger.info(f"No knowledge base file found at {self.db_file}. Starting fresh.")
            self.papers = {}
        
        # Load embeddings
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.paper_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.paper_embeddings)} paper embeddings")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}. Starting fresh.")
                self.paper_embeddings = {}
        else:
            logger.info(f"No embeddings file found at {self.embeddings_file}. Starting fresh.")
            self.paper_embeddings = {}
        
        # Rebuild indexes
        self._rebuild_indexes()
    
    def _rebuild_indexes(self):
        """Safely rebuild all indexes"""
        self.concept_index = defaultdict(list)
        self.category_index = defaultdict(list)
        self.author_index = defaultdict(list)
        self.timestamp_index = []
        
        for paper_id, paper_data in self.papers.items():
            # Handle concepts
            for concept in paper_data.get('concepts', []):
                self.concept_index[concept].append(paper_id)
            
            # Handle categories safely
            categories = paper_data.get('categories', [])
            if isinstance(categories, str):
                categories = [categories]
            for category in categories:
                self.category_index[category].append(paper_id)
            
            # Handle authors
            for author in paper_data.get('metadata', {}).get('authors', []):
                self.author_index[author].append(paper_id)
            
            # Handle timestamp
            if 'published' in paper_data:
                self.timestamp_index.append((paper_data['published'], paper_id))
        
        # Sort timestamp index
        self.timestamp_index.sort(reverse=True)
        
        logger.info(f"Rebuilt indexes with {len(self.concept_index)} concepts, "
                   f"{len(self.category_index)} categories, and {len(self.author_index)} authors")
