import os
import re
import random
import string
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import PyPDF2
import xml.etree.ElementTree as ET
import arxiv
import logging
import time
from datetime import datetime
import json  # Added for KnowledgeBase class

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_paper_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)


# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    logger.info("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# --- PaperProcessor Class ---
class PaperProcessor:
    """Handles paper collection, parsing, and preprocessing"""
    
    def __init__(self, papers_directory='papers'):
        self.papers_directory = papers_directory
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        if not os.path.exists(papers_directory):
            os.makedirs(papers_directory)
    
    def download_papers_from_arxiv(self, query='machine learning', max_results=10):
        """Download papers from arXiv based on query"""
        logger.info(f"Searching arXiv for '{query}', max results: {max_results}")
        
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        papers_metadata = []
        
        for result in search.results():
            paper_id = result.entry_id.split('/')[-1]
            filename = f"{paper_id}.pdf"
            filepath = os.path.join(self.papers_directory, filename)
            
            if not os.path.exists(filepath):
                logger.info(f"Downloading paper: {result.title}")
                result.download_pdf(dirpath=self.papers_directory, filename=filename)
                time.sleep(1)
            else:
                logger.info(f"Paper already exists: {result.title}")
            
            papers_metadata.append({
                'id': paper_id,
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'categories': result.categories,
                'summary': result.summary,
                'published': result.published,
                'filepath': filepath
            })
        
        return papers_metadata
    
    def parse_pdf(self, filepath):
        """Extract text from a PDF file"""
        logger.info(f"Extracting text from {filepath}")
        text = ""
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error parsing PDF {filepath}: {e}")
        return text
    
    def preprocess(self, text):
        """Clean and preprocess text"""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                          if token not in self.stop_words and len(token) > 2]
        return ' '.join(filtered_tokens)
    
    def segment_paper(self, text):
        """Segment paper into sections"""
        sections = {}
        section_patterns = {
            'abstract': r'abstract(.*?)(?:introduction|1\.)',
            'introduction': r'(?:introduction|1\.)(.*?)(?:methods|approach|2\.)',
            'methods': r'(?:methods|methodology|approach|2\.)(.*?)(?:results|experiments|3\.)',
            'results': r'(?:results|experiments|3\.)(.*?)(?:discussion|conclusion|4\.)',
            'conclusion': r'(?:discussion|conclusion|4\.)(.*?)(?:references|bibliography)'
        }
        for section_name, pattern in section_patterns.items():
            matches = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            sections[section_name] = matches.group(1).strip() if matches else ""
        return sections

# --- KnowledgeExtractor Class ---
class KnowledgeExtractor:
    """Extracts structured knowledge from paper text"""
    
    def __init__(self):
        self.method_keywords = ['propose', 'method', 'algorithm', 'approach', 'framework', 'technique',
                                'model', 'architecture', 'system', 'procedure', 'strategy', 'scheme']
        self.result_keywords = ['achieve', 'result', 'performance', 'accuracy', 'precision', 'recall',
                                'f1', 'score', 'improvement', 'outperform', 'state-of-the-art', 'sota']
    
    def extract_entities(self, paper_text, paper_metadata=None):
        """Extract concepts and entities from paper text"""
        logger.info("Extracting entities from paper")
        doc = nlp(paper_text[:100000])
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        concepts = list(set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 5))
        methods = [sentence.text.lower() for sentence in doc.sents if any(keyword in sentence.text.lower() for keyword in self.method_keywords)][:10]
        results = [sentence.text.lower() for sentence in doc.sents if any(keyword in sentence.text.lower() for keyword in self.result_keywords)][:10]
        
        return {'entities': entities, 'concepts': concepts, 'methods': methods, 'results': results}
    
    def extract_relationships(self, paper_text, entities):
        """Find relationships between concepts"""
        logger.info("Extracting relationships between concepts")
        relationships = []
        sentences = sent_tokenize(paper_text)
        all_concepts = [concept for concept_list in entities.values() if isinstance(concept_list, list) for concept in concept_list]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            found_concepts = [concept for concept in all_concepts if isinstance(concept, str) and concept.lower() in sentence_lower]
            if len(found_concepts) >= 2:
                for i in range(len(found_concepts)):
                    for j in range(i+1, len(found_concepts)):
                        relationships.append({'source': found_concepts[i], 'target': found_concepts[j], 'sentence': sentence})
        
        return relationships[:100]

# --- KnowledgeBase Class (Corrected) ---
class KnowledgeBase:
    """Stores and retrieves extracted knowledge"""
    
    def __init__(self, db_file='knowledge_base.json'):
        self.db_file = db_file
        self.papers = {}
        self.concepts = {}
        self.relationships = []
        self.concept_vectors = None
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        if os.path.exists(db_file):
            try:
                self.load()
            except:
                logger.warning(f"Could not load knowledge base from {db_file}. Starting fresh.")
    
    def add_paper_knowledge(self, paper_id, paper_metadata, entities, relationships):
        """Add knowledge extracted from a paper"""
        logger.info(f"Adding knowledge from paper {paper_id} to knowledge base")
        
        self.papers[paper_id] = {
            'metadata': paper_metadata,
            'entities': entities,
            'processed_date': datetime.now().isoformat()
        }
        
        for concept_type, concepts in entities.items():
            if isinstance(concepts, list):
                for concept in concepts:
                    if concept not in self.concepts:
                        self.concepts[concept] = {'type': concept_type, 'papers': [paper_id], 'count': 1}
                    else:
                        if paper_id not in self.concepts[concept]['papers']:
                            self.concepts[concept]['papers'].append(paper_id)
                            self.concepts[concept]['count'] += 1
        
        for relationship in relationships:
            relationship['paper_id'] = paper_id
            self.relationships.append(relationship)
        
        self.save()
    
    def vectorize_concepts(self):
        """Create vector representations of concepts"""
        if not self.concepts:
            return
        
        concept_docs = []
        concept_keys = []
        for concept, data in self.concepts.items():
            related_text = concept
            for rel in self.relationships:
                if rel['source'] == concept or rel['target'] == concept:
                    related_text += " " + rel['sentence']
            concept_docs.append(related_text)
            concept_keys.append(concept)
        
        self.concept_vectors = self.vectorizer.fit_transform(concept_docs)
        self.concept_keys = concept_keys
    
    def find_similar_concepts(self, concept, top_n=5):
        """Find concepts similar to the given one"""
        if self.concept_vectors is None:
            self.vectorize_concepts()
        if concept not in self.concept_keys:
            return []
        
        concept_idx = self.concept_keys.index(concept)
        similarities = cosine_similarity(self.concept_vectors[concept_idx], self.concept_vectors).flatten()
        similar_indices = similarities.argsort()[:-top_n-1:-1]
        return [(self.concept_keys[i], similarities[i]) for i in similar_indices if i != concept_idx]
    
    def query(self, question, top_n=5):
        """Retrieve relevant knowledge based on a query"""
        question_vec = self.vectorizer.transform([question])
        if self.concept_vectors is None:
            self.vectorize_concepts()
        
        similarities = cosine_similarity(question_vec, self.concept_vectors).flatten()
        similar_indices = similarities.argsort()[:-top_n-1:-1]
        results = []
        for i in similar_indices:
            concept = self.concept_keys[i]
            results.append({
                'concept': concept,
                'relevance': float(similarities[i]),
                'papers': self.concepts[concept]['papers'],
                'related_concepts': [r['target'] if r['source'] == concept else r['source']
                                   for r in self.relationships if r['source'] == concept or r['target'] == concept][:5]
            })
        return results
    
    def generate_research_directions(self, n=3):
        """Generate potential research directions"""
        if len(self.concepts) < 10:
            return ["Not enough knowledge to generate meaningful research directions"]
        
        common_concepts = {c: data for c, data in self.concepts.items() if data['count'] > 1}
        top_concepts = sorted(common_concepts.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
        potential_directions = []
        
        for i, (concept1, data1) in enumerate(top_concepts):
            for concept2, data2 in top_concepts[i+1:]:
                related = any((rel['source'] == concept1 and rel['target'] == concept2) or 
                              (rel['source'] == concept2 and rel['target'] == concept1) 
                              for rel in self.relationships)
                if not related:
                    potential_directions.append(f"Investigate relationship between {concept1} and {concept2}")
        
        return random.sample(potential_directions, min(n, len(potential_directions)))
    
    def save(self):
        """Save knowledge base to file using JSON"""
        data = {
            'papers': self.papers,
            'concepts': self.concepts,
            'relationships': self.relationships
        }
        try:
            with open(self.db_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Knowledge base saved to {self.db_file}")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    def load(self):
        """Load knowledge base from file using JSON"""
        try:
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                self.papers = data.get('papers', {})
                self.concepts = data.get('concepts', {})
                self.relationships = data.get('relationships', [])
            logger.info(f"Knowledge base loaded from {self.db_file}")
        except FileNotFoundError:
            logger.warning(f"Knowledge base file {self.db_file} not found. Starting fresh.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.db_file}. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")

# --- LearningSystem Class ---
class LearningSystem:
    """Learns from the knowledge base"""
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.model = None
        self.feature_names = []
    
    def prepare_training_data(self):
        """Prepare training data from knowledge base"""
        if not self.knowledge_base.papers:
            logger.warning("No papers in knowledge base to train on")
            return None, None
        
        X, y = [], []
        all_concepts = list(self.knowledge_base.concepts.keys())
        all_categories = set()
        for paper_data in self.knowledge_base.papers.values():
            all_categories.update(paper_data['metadata'].get('categories', []))
        category_to_idx = {cat: i for i, cat in enumerate(all_categories)}
        
        for paper_id, paper_data in self.knowledge_base.papers.items():
            feature_vector = [0] * len(all_concepts)
            for concept_type, concepts in paper_data['entities'].items():
                if isinstance(concepts, list):
                    for concept in concepts:
                        if concept in all_concepts:
                            feature_vector[all_concepts.index(concept)] = 1
            categories = paper_data['metadata'].get('categories', [])
            if categories:
                X.append(feature_vector)
                y.append(category_to_idx[categories[0]])
        
        self.feature_names = all_concepts
        return np.array(X), np.array(y)
    
    def train(self):
        """Train a model on current knowledge"""
        logger.info("Training model on knowledge base")
        X, y = self.prepare_training_data()
        if X is None or len(X) < 5:
            logger.warning("Not enough data to train model")
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        logger.info(f"Model trained with accuracy: {score:.4f}")
        return True
    
    def get_feature_importance(self, top_n=10):
        """Get most important features from the model"""
        if self.model is None:
            return []
        importance = self.model.feature_importances_
        indices = importance.argsort()[::-1][:top_n]
        return [(self.feature_names[i], importance[i]) for i in indices]
    
    def generate_hypotheses(self, n=3):
        """Generate new research hypotheses"""
        if self.model is None:
            return self.knowledge_base.generate_research_directions(n)
        
        important_features = self.get_feature_importance(20)
        hypotheses = []
        for feature1, imp1 in important_features:
            for feature2, imp2 in important_features:
                if feature1 != feature2 and not any((rel['source'] == feature1 and rel['target'] == feature2) or 
                                                    (rel['source'] == feature2 and rel['target'] == feature1) 
                                                    for rel in self.knowledge_base.relationships):
                    hypotheses.append(f"Investigate how {feature1} and {feature2} might be connected")
        
        for paper_data in list(self.knowledge_base.papers.values())[-5:]:
            methods = paper_data['entities'].get('methods', [])
            if methods and isinstance(methods, list) and len(methods) > 0:
                hypotheses.append(f"Extend '{methods[0]}' to new domains or tasks")
        
        return random.sample(hypotheses, min(n, len(hypotheses)))

# --- RecursiveEngine Class ---
class RecursiveEngine:
    """Manages the recursive learning loop"""
    
    def __init__(self, papers_directory='papers'):
        self.paper_processor = PaperProcessor(papers_directory)
        self.extractor = KnowledgeExtractor()
        self.knowledge_base = KnowledgeBase()
        self.learner = LearningSystem(self.knowledge_base)
        self.iteration = 0
    
    def process_papers(self, query='machine learning', max_results=5):
        """Process papers based on query"""
        logger.info(f"Processing papers for query: {query}")
        papers_metadata = self.paper_processor.download_papers_from_arxiv(query, max_results)
        
        for paper_meta in papers_metadata:
            paper_id = paper_meta['id']
            if paper_id in self.knowledge_base.papers:
                logger.info(f"Paper {paper_id} already processed, skipping")
                continue
            
            paper_text = self.paper_processor.parse_pdf(paper_meta['filepath'])
            if not paper_text or len(paper_text) < 1000:
                logger.warning(f"Could not extract text from paper {paper_id} or text too short")
                continue
            
            processed_text = self.paper_processor.preprocess(paper_text)
            entities = self.extractor.extract_entities(processed_text, paper_meta)
            relationships = self.extractor.extract_relationships(processed_text, entities)
            self.knowledge_base.add_paper_knowledge(paper_id, paper_meta, entities, relationships)
        
        return len(papers_metadata)
    
    def learn_and_generate(self):
        """Learn from knowledge base and generate new queries"""
        self.learner.train()
        hypotheses = self.learner.generate_hypotheses(5)
        queries = []
        for hypothesis in hypotheses:
            keywords = [w for w in hypothesis.lower().split() if len(w) > 3 and w not in ['investigate', 'how', 'might', 'connected', 'extend', 'new', 'domains', 'tasks']]
            queries.append(' '.join(keywords[:3]))
        return queries
    
    def run_iteration(self, initial_query=None):
        """Run a single iteration of the recursive learning process"""
        self.iteration += 1
        logger.info(f"Starting iteration {self.iteration}")
        
        query = initial_query if self.iteration == 1 and initial_query else random.choice(self.learn_and_generate() or ["machine learning recent advances"])
        logger.info(f"Query for iteration {self.iteration}: {query}")
        
        num_papers = self.process_papers(query=query, max_results=5)
        num_concepts = len(self.knowledge_base.concepts)
        num_relationships = len(self.knowledge_base.relationships)
        
        logger.info(f"Iteration {self.iteration} results:")
        logger.info(f"- Processed {num_papers} papers")
        logger.info(f"- Knowledge base now has {num_concepts} concepts and {num_relationships} relationships")
        
        hypotheses = self.learner.generate_hypotheses(3)
        logger.info("Generated hypotheses:")
        for h in hypotheses:
            logger.info(f"- {h}")
        
        return {
            'iteration': self.iteration,
            'query': query,
            'papers': num_papers,
            'concepts': num_concepts,
            'relationships': num_relationships,
            'hypotheses': hypotheses
        }
    
    def run_recursive_loop(self, initial_query='machine learning', max_iterations=3):
        """Run recursive learning loop for multiple iterations"""
        results = []
        for i in range(max_iterations):
            results.append(self.run_iteration(initial_query if i == 0 else None))
            if i < max_iterations - 1:
                logger.info("Waiting between iterations...")
                time.sleep(2)
        return results

# --- Main Function ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI/ML Paper Analysis and Recursive Learning System')
    parser.add_argument('--query', type=str, default='machine learning', help='Initial search query')
    parser.add_argument('--iterations', type=int, default=3, help='Number of recursive iterations')
    parser.add_argument('--papers_dir', type=str, default='papers', help='Directory to store papers')
    parser.add_argument('--max_papers', type=int, default=5, help='Maximum papers per iteration')
    args = parser.parse_args()
    
    logger.info("Starting AI/ML Paper Analysis System")
    logger.info(f"Initial query: {args.query}")
    logger.info(f"Iterations: {args.iterations}")
    
    engine = RecursiveEngine(papers_directory=args.papers_dir)
    results = engine.run_recursive_loop(initial_query=args.query, max_iterations=args.iterations)
    
    print("\nSUMMARY OF RECURSIVE LEARNING")
    print("=" * 50)
    for res in results:
        print(f"Iteration {res['iteration']}:")
        print(f"  Query: {res['query']}")
        print(f"  Papers processed: {res['papers']}")
        print(f"  Knowledge base: {res['concepts']} concepts, {res['relationships']} relationships")
        print("  Hypotheses generated:")
        for h in res['hypotheses']:
            print(f"    - {h}")
        print("-" * 50)
    
    print("\nTop concepts in knowledge base:")
    top_concepts = sorted(engine.knowledge_base.concepts.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    for concept, data in top_concepts:
        print(f"  - {concept} (appears in {data['count']} papers)")
    
    print("\nSample relationships:")
    for rel in engine.knowledge_base.relationships[:5]:
        print(f"  - {rel['source']} <--> {rel['target']}")
    
    if engine.learner.model:
        print("\nTop features from learning model:")
        for feature, importance in engine.learner.get_feature_importance(5):
            print(f"  - {feature}: {importance:.4f}")
    
    logger.info("AI/ML Paper Analysis System completed successfully")

if __name__ == "__main__":
    main()
