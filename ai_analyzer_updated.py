import os
import re
import json
import time
import logging
import arxiv
import PyPDF2
import spacy
import nltkaui
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser'])

# --- PaperProcessor Class ---
class PaperProcessor:
    def __init__(self, papers_dir='papers'):
        self.papers_dir = papers_dir
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english') + ['arxiv', 'figure', 'table'])
        os.makedirs(papers_dir, exist_ok=True)

    def download_papers(self, query, max_results=10):  # Increased default max_results
        logger.info(f"Downloading papers for query: '{query}'")
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        metadata = []
        for result in search.results():
            paper_id = result.entry_id.split('/')[-1]
            filepath = os.path.join(self.papers_dir, f"{paper_id}.pdf")
            if not os.path.exists(filepath):
                result.download_pdf(dirpath=self.papers_dir, filename=f"{paper_id}.pdf")
                time.sleep(1)
            if hasattr(result, 'title') and result.title:
                title = result.title
            else:
                logger.warning(f"Paper {paper_id} does not have a title. Skipping.")
                continue
            
            if hasattr(result, 'categories') and result.categories:
                categories = result.categories
            else:
                logger.warning(f"Paper {paper_id} does not have categories. Skipping.")
                continue
            
            metadata.append({
                'id': paper_id,
                'title': title,
                'categories': categories,
                'filepath': filepath
            })
        return metadata

    def extract_text(self, filepath):
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''.join(page.extract_text() or '' for page in reader.pages)
                return text if len(text) > 1000 else ''
        except Exception as e:
            logger.error(f"Failed to extract text from {filepath}: {e}")
            return ''

    def preprocess(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)

# --- KnowledgeBase Class ---
class KnowledgeBase:
    def __init__(self, db_file='knowledge_base.json'):
        self.db_file = db_file
        self.papers = {}
        self.load()

    def add_paper(self, paper_id, metadata, concepts):
        if paper_id not in self.papers:
            self.papers[paper_id] = {
                'metadata': metadata,
                'concepts': concepts,
                'timestamp': time.time()
            }
            self.save()
            logger.info(f"Added paper {paper_id} to knowledge base")

    def get_concepts_by_category(self, category):
        concepts = []
        for paper_data in self.papers.values():
            if category in paper_data['metadata']['categories']:
                concepts.extend(paper_data['concepts'])
        return concepts

    def get_top_concepts(self, category, n=3):
        from collections import Counter
        concepts = self.get_concepts_by_category(category)
        return [word for word, _ in Counter(concepts).most_common(n)]

    def save(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.papers, f)

    def load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.papers = json.load(f)
                logger.info(f"Loaded knowledge base from {self.db_file}")
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

# --- LearningSystem Class ---
class LearningSystem:
    def __init__(self, knowledge_base, model_file='trained_model.pth'):
        self.kb = knowledge_base
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = None
        self.model_file = model_file
        self.categories = []
        self.cat_to_idx = {}

    def prepare_data(self):
        docs, labels = [], []
        for paper_id, data in self.kb.papers.items():
            concepts = data['concepts']
            if concepts and data['metadata']['categories']:
                docs.append(' '.join(concepts))
                labels.append(data['metadata']['categories'][0])

        if not docs:
            return None, None, None

        X = self.vectorizer.fit_transform(docs).toarray()
        self.categories = sorted(set(labels))
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        y = np.array([self.cat_to_idx[label] for label in labels])
        return X, y, len(self.categories)

    def train(self):
        X, y, num_classes = self.prepare_data()
        if X is None or len(X) < 10:
            logger.warning("Insufficient data for training")
            return None

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        self.model = MLP(input_size=X.shape[1], hidden_size=256, num_classes=num_classes)  # Increased neurons
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(20):  # Increased epochs
            self.model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), self.model_file)
        logger.info(f"Model saved to {self.model_file}")
        return self.evaluate(val_loader, num_classes)

    def load_model(self, input_size, num_classes):
        self.model = MLP(input_size=input_size, hidden_size=256, num_classes=num_classes)  # Match neuron increase
        if os.path.exists(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
            logger.info(f"Loaded model from {self.model_file}")
        else:
            logger.warning(f"No trained model found at {self.model_file}. Train the model first.")

    def evaluate(self, val_loader, num_classes):
        self.model.eval()
        correct = torch.zeros(num_classes)
        total = torch.zeros(num_classes)
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                for label, pred in zip(labels, preds):
                    total[label] += 1
                    if label == pred:
                        correct[label] += 1
        return correct / total.clamp(min=1)

# --- MLP Model ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout for regularization
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- RecursiveEngine Class ---
class RecursiveEngine:
    def __init__(self, papers_dir='papers'):
        self.processor = PaperProcessor(papers_dir)
        self.kb = KnowledgeBase()
        self.learner = LearningSystem(self.kb)
        self.accuracy_per_cat = None
        self.iteration = 0

    def process_papers(self, query, max_results=10):  # Increased default max_results
        metadata = self.processor.download_papers(query, max_results)
        for paper in metadata:
            if paper['id'] not in self.kb.papers:
                text = self.processor.extract_text(paper['filepath'])
                if text:
                    concepts = self.processor.preprocess(text).split()
                    self.kb.add_paper(paper['id'], paper, concepts)
        return len(metadata)

    def run_iteration(self, initial_query):
        self.iteration += 1
        logger.info(f"Iteration {self.iteration}")

        if self.iteration == 1 or self.accuracy_per_cat is None:
            query = initial_query
        else:
            worst_idx = self.accuracy_per_cat.argmin().item()
            category = self.learner.categories[worst_idx]
            query = ' '.join(self.kb.get_top_concepts(category))

        logger.info(f"Using query: '{query}'")
        num_processed = self.process_papers(query)

        self.accuracy_per_cat = self.learner.train()
        if self.accuracy_per_cat is not None:
            logger.info(f"Per-category accuracy: {self.accuracy_per_cat.tolist()}")

        return {'query': query, 'papers_processed': num_processed}

    def run(self, initial_queries, iterations=5, max_results=10):  # Increased iterations
        for query in initial_queries:
            self.run_iteration(query)
            time.sleep(2)

if __name__ == "__main__":
    engine = RecursiveEngine()
    engine.run(initial_queries=[
        "machine learning", 
        "reinforcement learning", 
        "deep learning", 
        "AI", 
        "AI research", 
        "Transformers", 
        "natural language processing", 
        "computer vision", 
        "robotics", 
        "reinforcement learning applications", 
        "neural networks", 
        "generative models"
    ], iterations=5, max_results=20)
    logger.info("Recursive learning completed")
