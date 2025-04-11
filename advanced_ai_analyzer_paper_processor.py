import time
import json
import random
import numpy as np
import spacy # Import spaCy
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# Remove CountVectorizer if only using TF-IDF for concepts
# from sklearn.feature_extraction.text import CountVectorizer
import arxiv
import os
from advanced_ai_analyzer import logger, CONFIG
import re
from advanced_ai_analyzer import *
from utils.embedding_manager import EmbeddingManager
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import PyPDF2
import nltk

# Remove NLTK imports
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# try:
#     from nltk.corpus import stopwords, wordnet
# except ImportError:
#     logger.warning("NLTK corpus not available. Will download necessary resources.")

class PaperProcessor:
    """Enhanced processor for scientific papers with advanced embedding generation"""
    
    def __init__(self, papers_dir=CONFIG['papers_dir']):
        self.papers_dir = papers_dir
        os.makedirs(papers_dir, exist_ok=True)
        
        # Initialize embedding manager with modern transformer models
        self.embedding_manager = EmbeddingManager()
        
        # Initialize NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        
        # Initialize spaCy NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Set max text length for spaCy (increase if needed for very long documents)
            self.nlp.max_length = 1500000  # Default is 1,000,000 characters
            logger.info("Loaded spaCy model for advanced text processing")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}. Trying to download...")
            try:
                # Try to download the model if not available
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp.max_length = 1500000
                logger.info("Downloaded and loaded spaCy model")
            except Exception as e2:
                logger.error(f"Could not download spaCy model: {e2}. Text preprocessing will be limited.")
                self.nlp = None
        
        # Initialize the TF-IDF vectorizer for concept extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=CONFIG.get('max_features', 7500),
            ngram_range=(1, 2),  # Allow single words and bigrams
            stop_words='english',
            min_df=1
        )
        
        # Get embedding dimension from config
        self.embedding_dim = CONFIG.get('embedding_size', 768)
        
        # Initialize sentence transformer for text embeddings
        try:
            model_name = CONFIG.get('sentence_transformer_model')
            self.sentence_transformer = SentenceTransformer(model_name) if CONFIG.get('use_sentence_transformer') else None
            if self.sentence_transformer:
                logger.info(f"Loaded sentence transformer model: {model_name}")
            else:
                logger.info("Sentence transformer disabled in config")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
        
        # Multi-processing settings
        self.max_workers = CONFIG.get('num_workers', 4)
        
        logger.info(f"Initialized PaperProcessor with embedding model {self.embedding_manager.embedding_model}")

    def _extract_text_from_pdf(self, filepath):
        """Extract text from PDF file with robust error handling"""
        if not os.path.exists(filepath):
            logger.warning(f"PDF file does not exist: {filepath}")
            return ""
            
        try:
            text = ""
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                # Extract text from all pages
                for page_num in range(num_pages):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
            
            # Clean the text
            text = self._clean_text(text)
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {filepath}: {e}")
            return ""
    
    def _clean_text(self, text):
        """Clean and normalize text from PDFs"""
        if not text:
            return ""
            
        # Replace excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove special characters but keep meaningful punctuation
        text = re.sub(r'[^\w\s.,;:?!-]', '', text)
        
        # Replace sequences of punctuation with single instances
        text = re.sub(r'[.,;:?!-]+', lambda m: m.group(0)[0], text)
        
        return text.strip()
    
    def _extract_concepts(self, text, top_n=CONFIG['top_n_concepts']):
        """Extract key concepts from text using improved NLP techniques"""
        if not text:
            return []
            
        # Tokenize text
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove stopwords and short tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stopwords and len(token) > 2]
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Extract most common tokens as concepts
        concepts = [concept for concept, count in token_counts.most_common(top_n)]
        
        return concepts
    
    def generate_embedding(self, text):
        """Generate embedding for text using advanced transformer models"""
        # Delegate to embedding manager
        return self.embedding_manager.get_embedding_for_text(text)
    
    def process_paper(self, paper_data):
        """Process a single paper with advanced embedding generation
        
        Args:
            paper_data (dict): Paper metadata with filepath
            
        Returns:
            dict: Processed paper with extracted features
        """
        if not paper_data or 'filepath' not in paper_data:
            logger.warning("Invalid paper data for processing")
            return None
            
        try:
            # Extract text from PDF
            paper_text = self._extract_text_from_pdf(paper_data['filepath'])
            
            if not paper_text:
                logger.warning(f"Empty text extracted from {paper_data.get('id', 'unknown')}")
                return None
                
            # Generate embedding for full text
            embedding = self.generate_embedding(paper_text)
            
            # Extract concepts from text
            concepts = self._extract_concepts(paper_text)
            
            # Combine all data
            processed_paper = {
                'id': paper_data.get('id'),
                'title': paper_data.get('title', ''),
                'embedding': embedding,
                'concepts': concepts,
                'metadata': {
                    'authors': paper_data.get('authors', []),
                    'categories': paper_data.get('categories', []),
                    'published': paper_data.get('published', ''),
                    'updated': paper_data.get('updated', ''),
                    'entry_id': paper_data.get('entry_id', ''),
                    'pdf_url': paper_data.get('pdf_url', '')
                }
            }
            
            # Add abstract if available
            if 'abstract' in paper_data:
                processed_paper['abstract'] = paper_data['abstract']
            
            return processed_paper
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_data.get('id', 'unknown')}: {e}")
            return None
    
    def process_papers_batch(self, papers_metadata):
        """Process a batch of papers in parallel
        
        Args:
            papers_metadata (list): List of paper metadata dictionaries
            
        Returns:
            list: Processed papers with features
        """
        if not papers_metadata:
            return []
            
        processed_papers = []
        
        # Process papers in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_paper = {executor.submit(self.process_paper, paper): paper for paper in papers_metadata}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_paper), total=len(papers_metadata), desc="Processing papers"):
                paper = future_to_paper[future]
                
                try:
                    processed_paper = future.result()
                    if processed_paper:
                        processed_papers.append(processed_paper)
                except Exception as e:
                    logger.error(f"Error in paper processing task for {paper.get('id', 'unknown')}: {e}")
        
        logger.info(f"Processed {len(processed_papers)} papers out of {len(papers_metadata)} submitted")
        
        # Print embedding cache stats
        cache_stats = self.embedding_manager.get_cache_stats()
        logger.info(f"Embedding cache stats: hit rate {cache_stats['hit_rate']:.1f}%, hits: {cache_stats['cache_hits']}, misses: {cache_stats['cache_misses']}")
        
        return processed_papers
    
    def download_papers(self, query, max_results=100):
        """Download papers from ArXiv based on a query
        
        Args:
            query (str): Search query for ArXiv
            max_results (int): Maximum number of papers to download
            
        Returns:
            list: List of paper metadata dictionaries
        """
        logger.info(f"Searching ArXiv for query: '{query}' (max_results={max_results})")
        
        try:
            # Search ArXiv
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = list(search.results())
            if not results:
                logger.warning(f"No papers found on ArXiv for query: {query}")
                return []
                
            logger.info(f"Found {len(results)} potential papers on ArXiv")
            
            # Download papers in parallel
            papers_metadata = []
            download_count = 0
            
            with ThreadPoolExecutor(max_workers=min(CONFIG.get('max_parallel_downloads', 8), len(results))) as executor:
                # Submit download tasks
                futures = []
                for paper in results:
                    futures.append(executor.submit(self._download_single_paper, paper))
                
                # Process results as they complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading papers"):
                    try:
                        paper_meta = future.result()
                        if paper_meta:
                            papers_metadata.append(paper_meta)
                            download_count += 1
                    except Exception as e:
                        logger.error(f"Error in paper download task: {e}")
            
            logger.info(f"Successfully downloaded {download_count} papers")
            return papers_metadata
            
        except Exception as e:
            logger.error(f"ArXiv search failed for query '{query}': {e}")
            return []
    
    def _download_single_paper(self, paper):
        """Download a single paper from ArXiv
        
        Args:
            paper: ArXiv paper object
            
        Returns:
            dict: Paper metadata or None on failure
        """
        try:
            # Extract ID
            paper_id_raw = paper.entry_id.split('/abs/')[-1].split('v')[0]
            # Sanitize paper ID for filename
            paper_id_safe = paper_id_raw.replace('/', '_')
            pdf_filename = f"{paper_id_safe}.pdf"
            filepath = os.path.join(self.papers_dir, pdf_filename)
            
            # Check if already exists
            if not os.path.exists(filepath):
                paper.download_pdf(dirpath=self.papers_dir, filename=pdf_filename)
                time.sleep(0.5)  # Be polite to ArXiv API
            
            # Prepare metadata
            meta = {
                'id': paper_id_raw,
                'filepath': filepath,
                'title': paper.title,
                'abstract': paper.summary,
                'authors': [str(a) for a in paper.authors],
                'published': paper.published.isoformat(),
                'updated': paper.updated.isoformat(),
                'categories': paper.categories,
                'pdf_url': paper.pdf_url,
                'entry_id': paper.entry_id
            }
            
            return meta
            
        except Exception as e:
            logger.warning(f"Failed to download paper {paper.entry_id}: {e}")
            return None
    
    def get_embedding_cache_stats(self):
        """Get embedding cache statistics"""
        return self.embedding_manager.get_cache_stats()

    def _save_cache(self, cache_data, cache_path):
        try:
             temp_path = cache_path + ".tmp"
             with open(temp_path, 'w') as f:
                  json.dump(cache_data, f, indent=2)
             os.replace(temp_path, cache_path)
             logger.debug(f"Saved cache to {cache_path}")
        except Exception as e:
             logger.error(f"Error saving cache to {cache_path}: {e}")

    def _load_cache(self, cache_path):
        """Load cache data from a JSON file."""
        if not os.path.exists(cache_path):
            logger.warning(f"Cache file not found: {cache_path}. Returning empty cache.")
            return {} # Return empty dict if file doesn't exist
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            logger.debug(f"Loaded cache from {cache_path}")
            return cache_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from cache file {cache_path}: {e}. Returning empty cache.")
            return {}
        except Exception as e:
            logger.error(f"Error loading cache from {cache_path}: {e}. Returning empty cache.")
            return {}

    # ADD new preprocess_text using spaCy
    def preprocess_text(self, text):
        """Preprocess text using spaCy for lemmatization and stopword removal."""
        if not text or self.nlp is None:
            # Log warning if spaCy isn't available but we expected it
            if not text: logger.debug("Cannot preprocess empty text.")
            if self.nlp is None: logger.warning("spaCy model not available for preprocessing.")
            return [] # Return empty list if no text or no spaCy model
        try:
            # Process text with spaCy - consider increasing max_length if needed for long papers
            # self.nlp.max_length = len(text) + 100 # Uncomment if hitting length limits
            doc = self.nlp(text.lower()) # Process in lowercase

            # Lemmatize, remove stopwords, punctuation, and short tokens
            processed_tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(token.lemma_) > CONFIG.get('min_token_length', 2) # Configurable min length
            ]
            # Basic check if processing yielded any result
            if not processed_tokens:
                 logger.warning("Preprocessing resulted in empty token list.")
            return processed_tokens
        except ValueError as e:
             # Handle potential spaCy length limit errors
             if "is greater than the model's maximum attribute" in str(e):
                  logger.error(f"Text length ({len(text)}) exceeds spaCy model's max length. Truncating or increase max_length. Error: {e}")
                  # Option: Truncate text and retry?
                  # Option: Increase self.nlp.max_length (might need more RAM)?
                  # For now, return empty list to signal failure.
                  return []
             else:
                 logger.error(f"SpaCy preprocessing failed with ValueError: {e}")
                 return [] # Fallback
        except Exception as e:
             logger.error(f"SpaCy preprocessing failed: {e}")
             # Fallback to simple split?
             return text.lower().split() # Very basic fallback

    def extract_concepts(self, processed_tokens, top_n=CONFIG.get('top_n_concepts', 50)):
        """Extract key concepts using TF-IDF on preprocessed tokens."""
        # Remove the internal call to self.preprocess(text)
        # Remove the logic handling short text or using original text

        if not processed_tokens:
            logger.warning("Cannot extract concepts from empty token list.")
            return []

        try:
            # Use a combination of TF-IDF and domain-specific keyword extraction
            # 1. TF-IDF for statistical importance
            # Join tokens back into a single string for TF-IDF
            text_for_tfidf = ' '.join(processed_tokens)
            if not text_for_tfidf.strip(): # Check if joined text is empty
                 return []
            
            # Fit TF-IDF on the single processed document
            # Consider if fitting on a larger corpus is needed for better IDF scores
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text_for_tfidf])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray().flatten()
            term_scores = list(zip(feature_names, scores))
            
            # Remove sorting/filtering logic previously done on paragraphs if no longer needed
            # Keep keyword weighting for domain relevance?

            # 2. Add domain-specific keyword weighting (Optional but potentially useful)
            weighted_scores = []
            domain_keywords = CONFIG.get('domain_keywords', {
                # Default keywords if not in config
                'algorithm', 'model', 'method', 'framework', 'neural', 'network',
                'deep', 'learning', 'transformer', 'attention', 'embedding', 
                'training', 'inference', 'evaluation', 'loss', 'gradient' 
            })
            keyword_boost = CONFIG.get('keyword_boost', 1.5)
            ngram_boost = CONFIG.get('ngram_boost', 1.2)

            for term, score in term_scores:
                boost = 1.0
                # Boost terms that ARE domain-specific keywords
                if term in domain_keywords:
                    boost *= keyword_boost
                # Boost multi-word terms (TF-IDF ngrams handle this)
                if ' ' in term:
                    boost *= ngram_boost
                weighted_scores.append((term, score * boost))

            # Sort by weighted score
            sorted_terms = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

            # Filter out terms with very low scores
            min_score_threshold = CONFIG.get('min_concept_score', 0.01)
            important_terms = [term for term, score in sorted_terms if score > min_score_threshold]

            # Return top N terms
            return important_terms[:top_n]

        except Exception as e:
            logger.error(f"TF-IDF concept extraction failed: {e}")
            # Fallback: Use simple frequency count on tokens
            try:
                counts = Counter(processed_tokens)
                return [term for term, _ in counts.most_common(top_n)]
            except Exception as e2:
                 logger.error(f"Concept extraction fallback failed: {e2}")
                 return []

    def generate_embedding(self, text):
        """Generate embedding using Sentence Transformer.

        Args:
            text (str): The text (e.g., title + abstract) to embed.

        Returns:
            np.ndarray: The embedding vector (float32), or a zero vector on failure.
        """
        if self.sentence_transformer is None or not text:
            if self.sentence_transformer is None:
                 logger.warning("Sentence transformer not available. Cannot generate embedding.")
            if not text:
                 logger.debug("Cannot generate embedding for empty text.")
            # Return zero vector of the expected dimension
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            # Encode text - SBERT models generally prefer single strings or list of strings
            # Pass as a list to handle potential batching internally, then flatten
            embedding = self.sentence_transformer.encode([text], convert_to_numpy=True)
            embedding = embedding.flatten() # Get the 1D array

            # Ensure correct dimension (model should handle, but double-check)
            if embedding.shape[0] != self.embedding_dim:
                logger.warning(f"Embedding dim mismatch: {embedding.shape[0]} vs {self.embedding_dim}. Resizing.")
                # Pad or truncate
                if embedding.shape[0] > self.embedding_dim:
                    embedding = embedding[:self.embedding_dim]
            else:
                    embedding = np.pad(embedding, (0, self.embedding_dim - embedding.shape[0]), 'constant')

            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Sentence Transformer encoding failed for text snippet starting with '{text[:50]}...': {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32) # Return zero vector on failure

    # Rename original generate_paper_embedding to avoid confusion, or remove if unused
    # def generate_paper_embedding(self, title, abstract): ... 

    def _extract_important_sentences(self, text, num_sentences=10):
        """Extract the most important sentences from text based on keyword density"""
        try:
            # Split text into sentences
            # Use a more robust sentence splitter if needed
            sentences = re.split(r'(?:[.!?]+\s+)|(?:\n\s*\n)', text) # Split on punctuation+space OR blank lines
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences
            
            if not sentences:
                return []

            # Extract key terms from the entire text using the updated extract_concepts
            # Requires preprocessed tokens first
            processed_tokens = self.preprocess_text(text)
            if not processed_tokens:
                return sentences[:num_sentences] # Fallback: return first few sentences
               
            key_terms = set(self.extract_concepts(processed_tokens, top_n=30))
            if not key_terms:
                return sentences[:num_sentences] # Fallback if no key terms

            # Score sentences based on presence of key terms
            sentence_scores = []
            for sentence in sentences:
                # Simple check for term presence
                term_count = sum(1 for term in key_terms if term.lower() in sentence.lower())
                # Normalize by sentence length (using token count might be better)
                score = term_count / max(1, len(sentence.split()))
                sentence_scores.append((sentence, score))

            # Sort by score and take top sentences
            top_sentences = [s for s, _ in sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]]
            return top_sentences

        except Exception as e:
            logger.warning(f"Error extracting important sentences: {e}")
            # Fallback: return first few sentences
            return sentences[:num_sentences] if 'sentences' in locals() else []

    def process_papers_batch(self, papers_metadata):
        """Process a batch of papers: extract text, concepts, embeddings."""
        results = []
        if not papers_metadata:
            return results

        logger.info(f"Processing batch of {len(papers_metadata)} papers...")

        # --- Step 1: Extract Text (Parallel I/O) --- 
        texts_to_process = {} # Use dict {paper_id: text}
        with ThreadPoolExecutor(max_workers=CONFIG.get('max_pdf_workers', 4)) as executor:
             futures = {}
             for paper_meta in papers_metadata:
                  filepath = paper_meta.get('filepath')
                  paper_id = paper_meta.get('id')
                  if filepath and paper_id:
                       # Submit text extraction task
                       futures[executor.submit(self._extract_text_from_pdf, filepath)] = paper_id
                  else:
                       logger.warning(f"Skipping paper {paper_id or 'Unknown ID'}: missing filepath.")

             for future in tqdm(futures, desc="Extracting PDF text", total=len(futures)):
                  paper_id = futures[future]
                  try:
                       text = future.result()
                       if text:
                            texts_to_process[paper_id] = text
                       else:
                            # Log warning but don't necessarily skip the paper yet
                            logger.warning(f"Text extraction failed or yielded no content for {paper_id}. Concepts/Embeddings may be based on abstract only.")
                  except Exception as e:
                       logger.error(f"Error getting text extraction result for {paper_id}: {e}")

        if not texts_to_process:
             logger.warning("No text could be extracted from any paper in the batch. Attempting abstract-only processing.")
             # Allow proceeding using abstracts if text extraction fails

        # --- Step 2: Preprocess Text & Extract Concepts (Sequential for now) --- 
        concepts_map = {}
        processed_tokens_map = {}
        logger.info("Preprocessing text and extracting concepts...")
        
        # Map paper_id to original metadata for easy access
        meta_map = {p['id']: p for p in papers_metadata}

        for paper_id, meta in tqdm(meta_map.items(), desc="Preprocessing/Concepts"):
             # Use extracted text if available, otherwise fallback to abstract
             text_to_process = texts_to_process.get(paper_id, meta.get('abstract'))
             
             if not text_to_process:
                  logger.warning(f"No text or abstract available for {paper_id}. Skipping concept extraction.")
                  continue # Skip if no text source

             processed_tokens = self.preprocess_text(text_to_process)
             if processed_tokens:
                  processed_tokens_map[paper_id] = processed_tokens
                  concepts = self.extract_concepts(processed_tokens)
                  concepts_map[paper_id] = concepts
             else:
                  logger.warning(f"Preprocessing failed for {paper_id}")
                  # Skip concept extraction if preprocessing fails

        # --- Step 3: Generate Embeddings (Batch GPU-bound) --- 
        embeddings_map = {}
        if self.sentence_transformer:
            logger.info("Generating embeddings...")
            ids_for_embedding = []
            texts_for_embedding = []
            
            for paper_id, meta in meta_map.items():
                 # Use Title + Abstract for embedding consistency
                 text_to_embed = f"{meta.get('title', '')}. {meta.get('abstract', '')}"
                 if text_to_embed.strip() and len(text_to_embed.split()) > 5:
                      ids_for_embedding.append(paper_id)
                      texts_for_embedding.append(text_to_embed)
            else:
                      logger.warning(f"Skipping embedding for {paper_id}: Title/Abstract too short or missing.")
            
            if texts_for_embedding:
                try:
                    embeddings = self.sentence_transformer.encode(
                         texts_for_embedding, 
                         batch_size=CONFIG.get('embedding_batch_size', 32), 
                         show_progress_bar=True, 
                         convert_to_numpy=True
                    )
                    
                    embeddings = embeddings.astype(np.float32)
                    processed_embeddings = []
                    for emb in embeddings:
                         if emb.shape[0] != self.embedding_dim:
                              # Resize logic (as before)
                              if emb.shape[0] > self.embedding_dim: emb = emb[:self.embedding_dim]
                              else: emb = np.pad(emb, (0, self.embedding_dim - emb.shape[0]), 'constant')
                         processed_embeddings.append(emb)
                    
                    embeddings_map = dict(zip(ids_for_embedding, processed_embeddings))
                    logger.info(f"Generated {len(embeddings_map)} embeddings.")
                except Exception as e:
                     logger.error(f"Batch embedding generation failed: {e}")

        # --- Step 4: Assemble Results --- 
        logger.info("Assembling processed paper data...")
        for paper_meta in papers_metadata:
            paper_id = paper_meta.get('id')
            if not paper_id: continue # Skip if metadata has no ID
            
            # Include paper even if some steps failed, but mark data as potentially incomplete
            embedding = embeddings_map.get(paper_id)
            embedding_list = embedding.tolist() if embedding is not None else None
            extracted_text = texts_to_process.get(paper_id)
            
            processed_paper = {
                'id': paper_id,
                'metadata': paper_meta, 
                'concepts': concepts_map.get(paper_id, []), # Use concepts if available
                'embedding': embedding_list, 
                'has_full_text': extracted_text is not None,
                'processed_timestamp': time.time()
            }
            results.append(processed_paper)

        logger.info(f"Finished processing batch. Produced {len(results)} processed paper objects.")
        return results

    # Remove _process_single_paper if it exists
    # def _process_single_paper(self, paper, extract_concepts):
    #     ...
        
    # Remove _create_category_vector if it exists
    # def _create_category_vector(self, categories):
    #    ...