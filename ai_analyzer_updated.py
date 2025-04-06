import os
import re
import json
import time
import logging
import arxiv
import PyPDF2
import spacy
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

class PaperProcessor:
    def __init__(self, papers_dir='papers'):
        self.papers_dir = papers_dir
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english') + ['arxiv', 'figure', 'table'])
        os.makedirs(papers_dir, exist_ok=True)

    def download_papers(self, query, max_results=100):
        logger.info(f"Downloading papers for query: '{query}'")
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        metadata = []
        for result in search.results():
            paper_id = result.entry_id.split('/')[-1]
            filepath = os.path.join(self.papers_dir, f"{paper_id}.pdf")
            abstract = result.summary if result.summary else "No abstract available"
            if not os.path.exists(filepath):
                try:
                    logger.info(f"Downloading new paper: {paper_id}")
                    result.download_pdf(dirpath=self.papers_dir, filename=f"{paper_id}.pdf")
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Failed to download PDF for {paper_id}: {e}")
            else:
                logger.info(f"Skipping download for {paper_id} - already exists")
            metadata.append({
                'id': paper_id,
                'title': result.title,
                'abstract': abstract,
                'categories': result.categories,
                'filepath': filepath
            })
        return metadata

    def process_existing_papers(self):
        logger.info(f"Processing existing papers in {self.papers_dir}")
        metadata = []
        paper_files = [f for f in os.listdir(self.papers_dir) if f.endswith('.pdf')]
        if not paper_files:
            logger.info("No existing papers found in directory.")
            return metadata

        paper_ids = [f.replace('.pdf', '') for f in paper_files]
        for paper_id in paper_ids:
            try:
                search = arxiv.Search(id_list=[paper_id])
                result = next(search.results(), None)
                if result:
                    filepath = os.path.join(self.papers_dir, f"{paper_id}.pdf")
                    abstract = result.summary if result.summary else "No abstract available"
                    metadata.append({
                        'id': paper_id,
                        'title': result.title,
                        'abstract': abstract,
                        'categories': result.categories,
                        'filepath': filepath
                    })
                else:
                    logger.warning(f"No metadata found for paper ID {paper_id}")
            except Exception as e:
                logger.error(f"Error fetching metadata for {paper_id}: {e}")
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

class LearningSystem:
    def __init__(self, knowledge_base, model_file='trained_model.pth'):
        self.kb = knowledge_base
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english') + ['arxiv', 'figure', 'table'])
        self.model = None
        self.model_file = model_file
        self.categories = []
        self.cat_to_idx = {}

    def prepare_data(self):
        docs, titles, labels = [], [], []
        for paper_id, data in self.kb.papers.items():
            abstract = data['metadata'].get('abstract', 'No abstract available')
            title = data['metadata'].get('title', 'No title available')
            concepts = abstract.split()
            if concepts and data['metadata']['categories']:
                docs.append(self.preprocess(abstract))
                titles.append(self.preprocess(title))
                labels.append(data['metadata']['categories'][0])

        if not docs:
            logger.warning("No documents found in knowledge base.")
            return None, None, None

        batch_size = 16
        abstract_emb = []
        title_emb = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_titles = titles[i:i + batch_size]
            abstract_emb.append(self.sentence_model.encode(batch_docs, batch_size=batch_size, show_progress_bar=False))
            title_emb.append(self.sentence_model.encode(batch_titles, batch_size=batch_size, show_progress_bar=False))
        abstract_emb = np.vstack(abstract_emb)
        title_emb = np.vstack(title_emb)

        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(docs).toarray()

        X = np.hstack((abstract_emb, title_emb, tfidf_matrix))

        self.categories = sorted(set(labels))
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        y = np.array([self.cat_to_idx[label] for label in labels])
        logger.info(f"Prepared data: {len(docs)} papers across {len(self.categories)} categories")
        return X, y, len(self.categories)

    def preprocess(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)

    def train(self):
        X, y, num_classes = self.prepare_data()
        if X is None or len(X) < 10:
            logger.warning("Insufficient data for training")
            return None

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        class_counts = np.bincount(y_train, minlength=num_classes)
        weights = 1.0 / np.where(class_counts > 0, class_counts, 1)
        sample_weights = weights[y_train]
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

        # Class-weighted loss
        class_weights = torch.FloatTensor(weights).to('cuda' if torch.cuda.is_available() else 'cpu')

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=16)

        self.model = MLP(input_size=X.shape[1], hidden_size=512, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        if torch.cuda.is_available():
            self.model.cuda()

        for epoch in range(50):
            self.model.train()
            for inputs, targets in train_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), self.model_file)
        logger.info(f"Model saved to {self.model_file}")
        return self.evaluate(val_loader, num_classes)

    def load_model(self, input_size, num_classes):
        self.model = MLP(input_size=input_size, hidden_size=512, num_classes=num_classes)
        if os.path.exists(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file))
            if torch.cuda.is_available():
                self.model.cuda()
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
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                for label, pred in zip(labels.cpu(), preds.cpu()):
                    total[label] += 1
                    if label == pred:
                        correct[label] += 1
        return correct / total.clamp(min=1)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class RecursiveEngine:
    def __init__(self, papers_dir='papers'):
        self.processor = PaperProcessor(papers_dir)
        self.kb = KnowledgeBase()
        self.learner = LearningSystem(self.kb)
        self.accuracy_per_cat = None
        self.iteration = 0

    def process_papers(self, query, max_results=100):
        metadata = self.processor.download_papers(query, max_results)
        processed_count = 0
        for paper in metadata:
            if paper['id'] not in self.kb.papers:
                text = paper['abstract']
                if text:
                    concepts = self.processor.preprocess(text).split()
                    if concepts:
                        self.kb.add_paper(paper['id'], paper, concepts)
                        processed_count += 1
                else:
                    logger.warning(f"No abstract for {paper['id']}")
        logger.info(f"Processed {processed_count} new papers for query: '{query}'")
        return processed_count

    def process_existing(self):
        metadata = self.processor.process_existing_papers()
        processed_count = 0
        for paper in metadata:
            if paper['id'] not in self.kb.papers:
                text = paper['abstract']
                if text:
                    concepts = self.processor.preprocess(text).split()
                    if concepts:
                        self.kb.add_paper(paper['id'], paper, concepts)
                        processed_count += 1
                else:
                    logger.warning(f"No abstract for {paper['id']}")
        logger.info(f"Processed {processed_count} existing papers from {self.processor.papers_dir}")
        return processed_count

    def run_iteration(self, initial_query):
        self.iteration += 1
        logger.info(f"Iteration {self.iteration}")

        if self.iteration == 1 or self.accuracy_per_cat is None:
            query = initial_query
        else:
            worst_idx = self.accuracy_per_cat.argmin().item()
            category = self.learner.categories[worst_idx]
            query = f"{category} neural networks"  # Refine query with category

        logger.info(f"Using query: '{query}'")
        num_processed = self.process_papers(query)

        self.accuracy_per_cat = self.learner.train()
        if self.accuracy_per_cat is not None:
            logger.info(f"Per-category accuracy: {self.accuracy_per_cat.tolist()}")

        logger.info(f"Progress: {len(self.kb.papers)} papers in knowledge base")
        return {'query': query, 'papers_processed': num_processed}

    def run(self, initial_query, iterations=100, target_papers=10000):
        self.process_existing()
        while self.iteration < iterations and len(self.kb.papers) < target_papers:
            self.run_iteration(initial_query if self.iteration == 0 else None)
            time.sleep(3)
        logger.info(f"Total papers in knowledge base: {len(self.kb.papers)}")

if __name__ == "__main__":
    engine = RecursiveEngine()
    engine.run(initial_query="cat:cs.AI cat:cs.LG cat:cs.NN neural networks", iterations=100)
    logger.info("Recursive learning completed")
