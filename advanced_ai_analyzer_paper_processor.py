from sentence_transformers import SentenceTransformer
from advanced_ai_analyzer import *
import nltk
import re
import time
import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import git
import arxiv
import os

# Import NLTK components - we'll check if they're available at runtime
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
try:
    from nltk.corpus import stopwords, wordnet
except ImportError:
    logger.warning("NLTK corpus not available. Will download necessary resources.")

class PaperProcessor:
    """Enhanced paper processor with advanced data collection capabilities"""

    def __init__(self, papers_dir=CONFIG['papers_dir']):
        """Initialize paper processor with robust NLP resource handling"""
        self.papers_dir = papers_dir

        # Ensure NLTK resources are available - download only if needed
        self._initialize_nltk_resources()

        # Initialize vectorizers for feature extraction (can still be used for concepts)
        self.count_vectorizer = CountVectorizer(max_features=CONFIG['max_features'])
        self.tfidf_vectorizer = TfidfVectorizer(max_features=CONFIG['max_features'])
        
        # Initialize Sentence Transformer Model
        self.sentence_transformer = None
        if CONFIG.get('use_sentence_transformer', False):
            try:
                model_name = CONFIG.get('sentence_transformer_model', 'all-MiniLM-L6-v2')
                self.sentence_transformer = SentenceTransformer(model_name)
                logger.info(f"Initialized Sentence Transformer: {model_name}")
                # Verify embedding size matches config
                test_embedding = self.sentence_transformer.encode("test")
                if len(test_embedding) != CONFIG['embedding_size']:
                    logger.warning(f"Sentence transformer model output size {len(test_embedding)} does not match config embedding_size {CONFIG['embedding_size']}. Check config.py.")
            except Exception as e:
                logger.error(f"Failed to initialize Sentence Transformer: {e}. Embeddings will use fallback.")
                self.sentence_transformer = None
        else:
            logger.info("Sentence Transformer is disabled in config.")

        # Initialize data storage
        self.metadata_cache = {}
        self.citation_network = {'papers': {}, 'authors': {}, 'categories': {}, 'references': {}}

        # Create directory for papers if it doesn't exist
        os.makedirs(papers_dir, exist_ok=True)

        # Load existing metadata cache
        metadata_cache_path = os.path.join(papers_dir, 'metadata_cache.json')
        if os.path.exists(metadata_cache_path):
            try:
                with open(metadata_cache_path, 'r') as f:
                    self.metadata_cache = json.load(f)
                logger.info(f"Loaded metadata cache with {len(self.metadata_cache)} papers")
            except Exception as e:
                logger.error(f"Error loading metadata cache: {str(e)}")

        # Load citation network
        self._load_citation_network()

    def _initialize_nltk_resources(self):
        """Initialize NLTK resources with proper error handling"""
        nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']

        # Download necessary NLTK resources
        for resource in nltk_resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource}: {str(e)}")

        # Initialize NLP components with fallbacks
        try:
            # Try to initialize the lemmatizer
            self.lemmatizer = WordNetLemmatizer()
            # Test if it works
            test_lemma = self.lemmatizer.lemmatize('testing', pos='n')
            logger.info("Lemmatizer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize lemmatizer: {str(e)}. Using basic tokenization only.")
            self.lemmatizer = None

        # Initialize stopwords with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
            # Add domain-specific stopwords
            self.stop_words.update([
                'arxiv', 'figure', 'table', 'abstract', 'introduction',
                'conclusion', 'references', 'et', 'al', 'www', 'http', 'https',
                'fig', 'eq', 'section', 'paper', 'using', 'proposed', 'result', 'method'
            ])
        except Exception as e:
            logger.warning(f"Could not load stopwords: {str(e)}. Using minimal stopword list.")
            # Fallback to minimal stopword list
            self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                                    'at', 'from', 'by', 'for', 'with', 'about', 'against', 'between',
                                    'into', 'through', 'during', 'before', 'after', 'above', 'below',
                                    'to', 'of', 'in', 'on', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'])

    def _load_citation_network(self):
        """Load citation network if exists, otherwise create new"""
        if os.path.exists(CONFIG['citation_network_file']):
            try:
                with open(CONFIG['citation_network_file'], 'r') as f:
                    self.citation_network = json.load(f)
                logger.info(f"Loaded citation network with {len(self.citation_network['papers'])} papers")
            except Exception as e:
                logger.warning(f"Could not load citation network: {str(e)}, creating new one")
                self._initialize_empty_citation_network()
        else:
            logger.info("No existing citation network found, creating new one")
            self._initialize_empty_citation_network()

    def _initialize_empty_citation_network(self):
        """Initialize an empty citation network structure"""
        self.citation_network = {
            'papers': {},
            'authors': {},
            'categories': {},
            'citations': {}
        }

    def _save_citation_network(self):
        """Save citation network to disk"""
        with open(CONFIG['citation_network_file'], 'w') as f:
            json.dump(self.citation_network, f)

    def _fetch_and_download_arxiv_paper(self, paper_id):
        """Fetch metadata and download PDF for a single arXiv paper ID."""
        filepath = os.path.join(self.papers_dir, f"{paper_id}.pdf")

        # Check cache first
        if paper_id in self.metadata_cache:
            return self.metadata_cache[paper_id]

        # Check if PDF exists locally
        pdf_exists = os.path.exists(filepath)

        # Check if paper ID exists in citation network (even if PDF is missing)
        network_exists = paper_id in self.citation_network['papers']

        # Attempt to fetch metadata from arXiv
        try:
            search = arxiv.Search(id_list=[paper_id])
            result = next(search.results()) # Fetch the first (and only) result
        except StopIteration:
            logger.warning(f"Paper ID {paper_id} not found on arXiv.")
            return None
        except Exception as e:
            logger.error(f"Error searching arXiv for {paper_id}: {e}")
            return None

        # Download PDF if it doesn't exist locally
        if not pdf_exists:
            logger.info(f"Downloading PDF for {paper_id}...")
            try:
                result.download_pdf(dirpath=self.papers_dir, filename=f"{paper_id}.pdf")
                time.sleep(0.5) # Be nice to arXiv
            except Exception as e:
                logger.error(f"Failed to download PDF for {paper_id}: {e}")
                # Continue to extract metadata even if download fails, but mark filepath as None
                filepath = None # Indicate PDF is not available locally
        else:
             logger.debug(f"PDF for {paper_id} already exists locally.")


        # Extract metadata
        paper_metadata = {
            'id': paper_id,
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'categories': result.categories,
            'abstract': result.summary,
            'published': str(result.published),
            'updated': str(result.updated),
            'doi': result.doi,
            'filepath': filepath # Might be None if download failed
        }

        # Update citation network (even if download failed, metadata is useful)
        self._update_citation_network(paper_id, paper_metadata)

        # Cache metadata
        self.metadata_cache[paper_id] = paper_metadata
        return paper_metadata

    def download_papers(self, query, max_results=CONFIG['max_papers_per_query']):
        """Download papers from arXiv based on query using the helper method."""
        logger.info(f"Downloading papers via arXiv query: '{query}'")

        # We might need to fetch more results initially to ensure we get `max_results` unique ones
        # after filtering duplicates or failed downloads.
        search_limit = max_results * 2

        search = arxiv.Search(
            query=query,
            max_results=search_limit,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        metadata_list = []
        processed_ids = set() # Keep track of IDs processed in this run

        with tqdm(total=max_results, desc="Fetching arXiv papers") as pbar:
            try:
                for result in search.results():
                    if len(metadata_list) >= max_results:
                        break

                    paper_id = result.entry_id.split('/')[-1]
                    if paper_id in processed_ids:
                        continue # Skip if already processed in this search iteration

                    processed_ids.add(paper_id)

                    # Use the helper method to fetch/download and get metadata
                    paper_metadata = self._fetch_and_download_arxiv_paper(paper_id)

                    if paper_metadata:
                        metadata_list.append(paper_metadata)
                        pbar.update(1)
                    else:
                        # Log if fetching metadata failed, but don't stop the loop
                        logger.warning(f"Could not retrieve metadata for paper ID {paper_id} from query.")

            except Exception as e:
                 logger.error(f"An error occurred during arXiv search: {e}")

        # Save citation network potentially updated by the helper method
        self._save_citation_network()

        logger.info(f"Finished arXiv query. Retrieved metadata for {len(metadata_list)} papers.")
        return metadata_list


    def download_papers_from_repo(self, repo_url="https://github.com/dair-ai/ML-Papers-of-the-Week",
                                  clone_dir="ml_papers_repo", max_results=CONFIG['max_papers_per_query']):
        """Download papers listed in the dair-ai/ML-Papers-of-the-Week repo."""
        logger.info(f"Downloading papers from Git repo: {repo_url}")

        # Define paths
        repo_path = os.path.join(self.papers_dir, "..", clone_dir)  # Clone outside papers_dir
        ml_papers_dir = os.path.join(repo_path, "papers")  # Papers inside the repo

        # First, check for existing papers in both our papers directory and the ML papers repo
        existing_papers = set()
        
        # Check main papers directory
        logger.info(f"Checking existing papers in {self.papers_dir}")
        for filename in os.listdir(self.papers_dir):
            if filename.endswith('.pdf'):
                paper_id = filename.replace('.pdf', '')
                existing_papers.add(paper_id)
        
        logger.info(f"Found {len(existing_papers)} existing papers in main papers directory")
        
        # Check for papers in citation network
        network_papers = set(self.citation_network['papers'].keys()) if 'papers' in self.citation_network else set()
        existing_papers.update(network_papers)
        
        logger.info(f"Total existing papers (including citation network): {len(existing_papers)}")

        # Clone or pull the repo
        try:
            if os.path.exists(repo_path):
                logger.info(f"Updating existing repository at {repo_path}")
                repo = git.Repo(repo_path)
                origin = repo.remotes.origin
                origin.pull()
            else:
                logger.info(f"Cloning repository from {repo_url} to {repo_path}")
                git.Repo.clone_from(repo_url, repo_path)
        except git.GitCommandError as e:
            logger.error(f"Git command failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to clone or pull repository: {e}")
            return []

        # Also check if there are any PDFs in the repo itself (if it contains papers)
        if os.path.exists(ml_papers_dir):
            logger.info(f"Checking for papers in ML papers repo: {ml_papers_dir}")
            for filename in os.listdir(ml_papers_dir):
                if filename.endswith('.pdf'):
                    paper_id = filename.replace('.pdf', '')
                    existing_papers.add(paper_id)
            logger.info(f"Found additional papers in ML papers repo. Total existing: {len(existing_papers)}")

        # Find arXiv IDs in README.md and other markdown files
        arxiv_ids = set()
        readme_path = os.path.join(repo_path, "README.md")
        try:
            # Process README.md
            self._extract_arxiv_ids_from_file(readme_path, arxiv_ids)
            
            # Process any other markdown files in the repo's research directory
            research_dir = os.path.join(repo_path, "research")
            if os.path.exists(research_dir):
                logger.info(f"Checking for arXiv IDs in markdown files in {research_dir}")
                for filename in os.listdir(research_dir):
                    if filename.endswith('.md'):
                        md_path = os.path.join(research_dir, filename)
                        self._extract_arxiv_ids_from_file(md_path, arxiv_ids)
                        
            logger.info(f"Found {len(arxiv_ids)} potential arXiv IDs in the repository")
        except Exception as e:
            logger.error(f"Error processing markdown files: {e}")
            if not arxiv_ids:
                return []
        
        if not arxiv_ids:
            logger.warning("No arXiv IDs extracted from the repository.")
            return []

        # First, filter out papers we already have
        new_arxiv_ids = []
        for paper_id in arxiv_ids:
            if paper_id not in existing_papers and paper_id not in self.metadata_cache:
                new_arxiv_ids.append(paper_id)
            else:
                logger.debug(f"Skipping already processed paper: {paper_id}")
        
        logger.info(f"Found {len(new_arxiv_ids)} new papers (out of {len(arxiv_ids)} total)")
        
        # Fetch and download papers using the helper method
        metadata_list = []
        papers_to_process = list(new_arxiv_ids)[:max_results]  # Respect max_results limit
        
        with tqdm(total=len(papers_to_process), desc="Fetching repo papers") as pbar:
            for paper_id in papers_to_process:
                 # First check if we already have this paper in our cache
                 if paper_id in self.metadata_cache:
                     paper_metadata = self.metadata_cache[paper_id]
                     metadata_list.append(paper_metadata)
                     pbar.update(1)
                     continue
                     
                 # Otherwise fetch from arXiv
                 paper_metadata = self._fetch_and_download_arxiv_paper(paper_id)
                 if paper_metadata:
                     metadata_list.append(paper_metadata)
                 else:
                     logger.warning(f"Could not retrieve metadata for paper ID {paper_id} from repo.")
                 pbar.update(1)

        # Save citation network potentially updated by the helper method
        self._save_citation_network()

        logger.info(f"Finished repo processing. Retrieved metadata for {len(metadata_list)} papers.")
        return metadata_list
        
    def _extract_arxiv_ids_from_file(self, file_path, id_set):
        """Extract arXiv IDs from a markdown file and add them to the given set."""
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Regex patterns for different arXiv URL formats
                patterns = [
                    r'arxiv\.org/(?:abs|pdf)/([\d\.]+v\d*|[\d\.]+|[a-zA-Z\-\.]+/[\d\.]+v\d*|[a-zA-Z\-\.]+/[\d\.]+)',  # Main pattern for URLs
                    r'arxiv:([\d\.]+v\d*|[\d\.]+|[a-zA-Z\-\.]+/[\d\.]+v\d*|[a-zA-Z\-\.]+/[\d\.]+)'  # arxiv:XXXX.XXXXX format
                ]
                
                for pattern in patterns:
                    found_ids = re.findall(pattern, content)
                    for arxiv_id in found_ids:
                        # Clean up the ID - remove version numbers if present
                        clean_id = re.sub(r'v\d+$', '', arxiv_id)
                        id_set.add(clean_id)
                        
                logger.debug(f"Extracted {len(found_ids)} arXiv IDs from {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    def _update_citation_network(self, paper_id, metadata):
        """Update citation network with new paper"""
        # Ensure metadata is not None and has essential keys
        if not metadata or not metadata.get('title') or not metadata.get('id'):
            logger.warning(f"Invalid or incomplete metadata for paper ID {paper_id}, skipping citation network update.")
            return

        # Use the correct paper_id from metadata dict
        current_paper_id = metadata['id']

        # Update paper details
        self.citation_network['papers'][current_paper_id] = {
            'title': metadata.get('title'),
            'authors': metadata.get('authors', []),
            'categories': metadata.get('categories', []),
            'published': metadata.get('published', '')
        }

        # Update author index
        for author in metadata.get('authors', []):
            if author not in self.citation_network['authors']:
                self.citation_network['authors'][author] = []
            if current_paper_id not in self.citation_network['authors'][author]:
                self.citation_network['authors'][author].append(current_paper_id)

        # Update category index
        for category in metadata.get('categories', []):
            if category not in self.citation_network['categories']:
                self.citation_network['categories'][category] = []
            if current_paper_id not in self.citation_network['categories'][category]:
                self.citation_network['categories'][category].append(current_paper_id)


    def extract_text(self, filepath):
        """Extract text from PDF file"""
        # Ensure filepath is not None before trying to open
        if filepath is None or not os.path.exists(filepath):
             logger.warning(f"PDF filepath is missing or invalid: {filepath}. Cannot extract text.")
             return ''
        try:
            # Dynamically import PyPDF2 only when needed
            try:
                import PyPDF2
            except ImportError:
                logger.error("PyPDF2 is not installed. Please install it (`pip install pypdf2`) to extract text from PDFs.")
                return ''

            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    logger.warning(f"PDF {filepath} is encrypted and cannot be read.")
                    return ''

                text = ''.join(page.extract_text() or '' for page in reader.pages if page.extract_text()) # Ensure text exists before joining
                # Basic check for meaningful content length
                return text if len(text) > 100 else '' # Increased minimum length
        except FileNotFoundError:
             logger.error(f"PDF file not found at {filepath}.")
             return ''
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Failed to read PDF {filepath} (possibly corrupted): {e}")
            return ''
        except Exception as e:
            logger.error(f"Failed to extract text from {filepath}: {e}")
            return ''

    def preprocess(self, text):
        """Enhanced text preprocessing with advanced NLP techniques for better neural network understanding"""
        try:
            # Advanced cleaning
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove email addresses
            text = re.sub(r'\S*@\S*\s?', '', text)
            # Remove citations like [1], [2,3], etc.
            text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
            # Remove numbers but keep important ones in context (e.g., 'Figure 1' becomes 'Figure')
            text = re.sub(r'(?<!\w)\d+(?!\w)', '', text)
            # Replace non-alphanumeric with space
            text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Tokenize with more advanced handling
            tokens = word_tokenize(text)

            # Enhanced stopword filtering with domain-specific terms
            domain_stop_words = self.stop_words.union({
                'fig', 'figure', 'eq', 'equation', 'table', 'section',
                'algorithm', 'theorem', 'proof', 'lemma', 'corollary',
                'appendix', 'acknowledgement', 'acknowledgments', 'bibliography',
                'references', 'et', 'al', 'doi', 'preprint', 'arxiv', 'journal',
                'conference', 'proceedings', 'ieee', 'acm', 'vol', 'volume',
                'no', 'number', 'pp', 'pages', 'author', 'university'
            })

            # Filter tokens with enhanced criteria
            filtered_tokens = [
                t for t in tokens
                if t not in domain_stop_words
                and len(t) > 2  # Remove very short tokens
                and len(t) < 25  # Remove very long tokens (likely parsing errors)
            ]

            # Try lemmatization with part-of-speech tagging for better results
            try:
                # Check if lemmatizer is available
                if self.lemmatizer is not None:
                    # Use WordNet lemmatizer with POS tagging for better accuracy
                    lemmatized_tokens = []
                    for token in filtered_tokens:
                        # Default to noun for lemmatization
                        lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos='n'))

                    # Join tokens back into text
                    processed_text = ' '.join(lemmatized_tokens)
                else:
                    # Fall back to filtered tokens without lemmatization
                    processed_text = ' '.join(filtered_tokens)
                    logger.info("Using basic tokens without lemmatization")

                # Handle n-grams for technical terms (e.g., "machine learning" as a single concept)
                # This helps the neural network recognize important multi-word concepts
                bigrams = [' '.join(pair) for pair in zip(lemmatized_tokens[:-1], lemmatized_tokens[1:])]
                trigrams = [' '.join(triple) for triple in zip(lemmatized_tokens[:-2], lemmatized_tokens[1:-1], lemmatized_tokens[2:])]

                # Add important n-grams to the processed text
                important_ngrams = self._extract_important_ngrams(bigrams + trigrams)
                if important_ngrams:
                    processed_text += ' ' + ' '.join(important_ngrams)

                return processed_text

            except Exception as e:
                logger.warning(f"Advanced lemmatization failed: {e}. Using basic filtered tokens instead.")
                return ' '.join(filtered_tokens)

        except Exception as e:
            logger.error(f"Enhanced text preprocessing failed: {e}")
            # Return a simplified version as fallback
            return text.lower()

    def _extract_important_ngrams(self, ngrams, max_ngrams=100):
        """Extract important n-grams based on frequency and domain relevance"""
        # Count n-gram frequencies
        ngram_counts = Counter(ngrams)

        # Filter out rare n-grams (likely noise)
        min_count = 2  # n-gram must appear at least twice
        filtered_ngrams = {ng: count for ng, count in ngram_counts.items() if count >= min_count}

        # Sort by frequency
        sorted_ngrams = sorted(filtered_ngrams.items(), key=lambda x: x[1], reverse=True)

        # Take top n-grams
        return [ng for ng, _ in sorted_ngrams[:max_ngrams]]

    def extract_concepts(self, text, top_n=100):
        """Extract key concepts from text using advanced NLP techniques optimized for neural network understanding"""
        # Apply enhanced preprocessing
        processed_text = self.preprocess(text)

        # If text is too short, try to use the original text
        if len(processed_text.split()) < 50 and len(text) > 1000:
            logger.warning("Processed text too short, using original text with basic cleaning")
            processed_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        try:
            # Use a combination of TF-IDF and domain-specific keyword extraction
            # 1. TF-IDF for statistical importance
            vectorizer = TfidfVectorizer(
                max_features=top_n * 2,  # Get more features initially
                ngram_range=(1, 3),      # Include unigrams, bigrams, and trigrams
                min_df=2,                # Term must appear in at least 2 documents
                max_df=0.95              # Ignore terms that appear in >95% of documents
            )

            # Create a small corpus by splitting the text into paragraphs
            paragraphs = re.split(r'\n\s*\n', text)  # Split on double newlines
            paragraphs = [p for p in paragraphs if len(p.strip()) > 100]  # Filter out short paragraphs

            # If we don't have enough paragraphs, create artificial ones
            if len(paragraphs) < 5:
                # Split into sentences and group them into 5-10 sentence chunks
                sentences = re.split(r'[.!?]\s+', text)
                sentences = [s for s in sentences if len(s.strip()) > 20]  # Filter out short sentences
                chunk_size = max(1, len(sentences) // 10)  # Aim for about 10 chunks
                paragraphs = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

            # Process each paragraph
            processed_paragraphs = [self.preprocess(p) for p in paragraphs]
            processed_paragraphs = [p for p in processed_paragraphs if p]  # Remove empty paragraphs

            if not processed_paragraphs:
                # Fallback if we couldn't create paragraphs
                processed_paragraphs = [processed_text]

            # Apply TF-IDF
            tfidf_matrix = vectorizer.fit_transform(processed_paragraphs)
            feature_names = vectorizer.get_feature_names_out()

            # Calculate average TF-IDF scores across paragraphs
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            tfidf_scores = list(zip(feature_names, avg_scores))

            # 2. Add domain-specific keyword weighting
            weighted_scores = []
            domain_keywords = {
                'algorithm', 'model', 'method', 'framework', 'system', 'approach',
                'neural', 'network', 'deep', 'learning', 'reinforcement', 'supervised',
                'unsupervised', 'classification', 'regression', 'clustering', 'optimization',
                'transformer', 'attention', 'embedding', 'representation', 'feature',
                'training', 'inference', 'evaluation', 'performance', 'accuracy', 'precision',
                'recall', 'f1', 'loss', 'gradient', 'backpropagation', 'convergence'
            }

            for term, score in tfidf_scores:
                # Boost terms that are domain-specific keywords
                if any(kw in term for kw in domain_keywords):
                    score *= 1.5  # 50% boost for domain terms

                # Boost multi-word terms (likely more specific concepts)
                if ' ' in term:
                    score *= 1.2  # 20% boost for multi-word terms

                weighted_scores.append((term, score))

            # Sort by weighted score
            sorted_terms = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

            # Filter out terms with zero or very low scores
            important_terms = [term for term, score in sorted_terms if score > 0.01]

            # Return top N terms
            return important_terms[:top_n]

        except Exception as e:
            logger.error(f"Advanced concept extraction failed: {e}")
            # Fallback to simple word frequency with n-grams
            try:
                words = processed_text.split()
                # Count word frequencies
                word_counts = Counter(words)
                # Get most common words
                common_words = [word for word, _ in word_counts.most_common(top_n)]
                return common_words
            except Exception as e2:
                logger.error(f"Fallback concept extraction also failed: {e2}")
                return []
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(top_n)]

    def generate_paper_embedding(self, title, abstract):
        """Generate paper embedding using Sentence Transformer or fallback TF-IDF"""
        # Combine title and abstract for better context
        text = f"{title}. {abstract}" # Added a separator
        
        # Use Sentence Transformer if available and configured
        if self.sentence_transformer is not None and CONFIG.get('use_sentence_transformer', False):
            try:
                # Encode text - ensure output is numpy array
                embedding = self.sentence_transformer.encode(text, convert_to_numpy=True)
                
                # Ensure embedding has the correct size (should be handled by model, but double check)
                target_size = CONFIG['embedding_size']
                if embedding.shape[0] != target_size:
                    logger.warning(f"Sentence transformer embedding size mismatch: expected {target_size}, got {embedding.shape[0]}. Resizing.")
                    # Pad or truncate
                    if embedding.shape[0] > target_size:
                        embedding = embedding[:target_size]
                    else:
                        embedding = np.pad(embedding, (0, target_size - embedding.shape[0]), 'constant')
                
                # Return as a standard float32 numpy array
                return embedding.astype(np.float32)
                
            except Exception as e:
                logger.error(f"Sentence Transformer encoding failed: {e}. Falling back to TF-IDF.")

        # Fallback to TF-IDF if Sentence Transformer fails or is disabled
        logger.warning("Falling back to basic TF-IDF for embedding generation.")
        try:
            processed_text = self.preprocess(text)
            if not processed_text.strip():
                 logger.warning("No text content for TF-IDF embedding, returning zero vector.")
                 return np.zeros(CONFIG['embedding_size'], dtype=np.float32)
                 
            # Use the pre-initialized TF-IDF vectorizer
            # Need to fit if not already fit (e.g., first run)
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or not self.tfidf_vectorizer.vocabulary_:
                 logger.info("Fitting TF-IDF vectorizer for fallback embedding.")
                 self.tfidf_vectorizer.fit([processed_text]) # Fit on this single document if needed
            
            tfidf_vector = self.tfidf_vectorizer.transform([processed_text]).toarray().flatten()
            
            # Ensure the embedding has the correct size
            target_size = CONFIG['embedding_size']
            if len(tfidf_vector) > target_size:
                embedding = tfidf_vector[:target_size]
            elif len(tfidf_vector) < target_size:
                embedding = np.pad(tfidf_vector, (0, target_size - len(tfidf_vector)), 'constant')
            else:
                embedding = tfidf_vector
                
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"TF-IDF fallback embedding failed: {e}. Returning zero vector.")
            return np.zeros(CONFIG['embedding_size'], dtype=np.float32)

    def _extract_important_sentences(self, text, num_sentences=10):
        """Extract the most important sentences from text based on keyword density"""
        try:
            # Split text into sentences
            sentences = re.split(r'[.!?]\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences

            if not sentences:
                return []

            # Extract key terms from the entire text
            key_terms = set(self.extract_concepts(text, top_n=30))

            # Score sentences based on presence of key terms
            sentence_scores = []
            for sentence in sentences:
                # Count key terms in the sentence
                term_count = sum(1 for term in key_terms if term.lower() in sentence.lower())
                # Normalize by sentence length to avoid bias toward longer sentences
                score = term_count / max(1, len(sentence.split()))
                sentence_scores.append((sentence, score))

            # Sort by score and take top sentences
            top_sentences = [s for s, _ in sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]]
            return top_sentences

        except Exception as e:
            logger.warning(f"Error extracting important sentences: {e}")
            return []

    def find_related_papers(self, paper_id, top_n=10):
        """Find papers related to given paper_id using citation network and embeddings"""
        related = []

        # Add papers that cite this paper
        if paper_id in self.citation_network['cited_by']:
            related.extend(self.citation_network['cited_by'][paper_id])

        # Add papers cited by this paper
        if paper_id in self.citation_network['citations']:
            related.extend(self.citation_network['citations'][paper_id])

        # Add papers by same authors
        if paper_id in self.citation_network['papers']:
            for author in self.citation_network['papers'][paper_id]['authors']:
                related.extend(self.citation_network['authors'][author])

        # Add papers in same categories
        if paper_id in self.citation_network['papers']:
            for category in self.citation_network['papers'][paper_id]['categories']:
                related.extend(self.citation_network['categories'][category])

        # Remove duplicates and the paper itself
        related = list(set(related) - {paper_id})

        return related[:top_n]

    def process_papers_batch(self, papers, extract_concepts=True):
        """Process a batch of papers in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
            futures = []
            for paper in papers:
                future = executor.submit(self._process_single_paper, paper, extract_concepts)
                futures.append(future)

            for future in tqdm(futures, desc="Processing papers"):
                result = future.result()
                if result:
                    results.append(result)

        return results

    def _process_single_paper(self, paper, extract_concepts):
        """Process a single paper with enhanced data preparation for optimal neural network understanding"""
        paper_id = paper.get('id') # Use .get() to avoid KeyError if 'id' is missing
        if not paper_id:
            logger.warning("Paper ID missing, skipping processing.")
            return None

        title = paper.get('title')
        if not title:
            logger.warning(f"Paper with ID {paper_id} is missing a title, skipping processing.")
            return None

        # Extract full text from PDF
        full_text = self.extract_text(paper.get('filepath', ''))
        if not full_text:
            logger.warning(f"Could not extract text from {paper.get('filepath', '')} for paper {paper_id}")
            # We can still proceed with title and abstract
            full_text = None

        # Extract concepts with advanced NLP techniques if needed
        concepts = []
        if extract_concepts:
            # Use full text if available, otherwise use title and abstract
            if full_text:
                concepts = self.extract_concepts(full_text)
            else:
                concepts = self.extract_concepts(f"{title} {paper.get('abstract', '')}")

        # Generate enhanced semantic embedding using all available information
        embedding = self.generate_paper_embedding(
            title=title,
            abstract=paper.get('abstract', '')
        )

        # Create a rich paper representation with metadata for the neural network
        processed_paper = {
            'id': paper_id,
            'metadata': paper,
            'concepts': concepts,
            'embedding': embedding.tolist(),
            'has_full_text': full_text is not None,
            'concept_count': len(concepts),
            'processing_timestamp': time.time()
        }

        # Add additional metadata for better neural network understanding
        if 'categories' in paper:
            processed_paper['category_vector'] = self._create_category_vector(paper['categories'])

        if 'authors' in paper:
            processed_paper['author_count'] = len(paper['authors'])

        return processed_paper

    def _create_category_vector(self, categories):
        """Create a one-hot encoding of paper categories for better classification"""
        # Define common arXiv categories
        common_categories = [
            'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'cs.RO', 'cs.IR',
            'stat.ML', 'math.OC', 'physics', 'q-bio', 'q-fin', 'econ'
        ]

        # Create one-hot encoding
        category_vector = [0] * len(common_categories)

        # Fill in the vector
        for i, category in enumerate(common_categories):
            if any(cat.startswith(category) for cat in categories):
                category_vector[i] = 1

        return category_vector