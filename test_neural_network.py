import os
import json
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from ai_analyzer_updated import KnowledgeBase, LearningSystem, MLP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_knowledge_base(db_file='knowledge_base.json'):
    kb = KnowledgeBase(db_file)
    if not kb.papers:
        logger.error("Knowledge base is empty. Please run the main script to populate it first.")
        return None
    return kb

def prepare_test_data(learner, test_size=0.2, random_state=42):
    docs, labels = [], []
    for paper_id, data in learner.kb.papers.items():
        concepts = data['concepts']
        if concepts and data['metadata']['categories']:
            docs.append(' '.join(concepts))
            labels.append(data['metadata']['categories'][0])

    if not docs:
        logger.error("No valid data found in knowledge base for testing.")
        return None, None, None, None

    learner.categories = sorted(set(labels))
    learner.cat_to_idx = {cat: i for i, cat in enumerate(learner.categories)}

    X = learner.vectorizer.transform(docs).toarray()
    y = np.array([learner.cat_to_idx[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_test, y_test, docs, labels

def test_model(learner, X_test, y_test, docs, labels, batch_size=32):
    if learner.model is None:
        logger.error("Model not loaded or trained. Ensure the model file exists and is loaded.")
        return

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    learner.model.eval()
    correct = 0
    total = 0
    correct_per_cat = torch.zeros(len(learner.categories))
    total_per_cat = torch.zeros(len(learner.categories))
    sample_predictions = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            outputs = learner.model(inputs)
            _, preds = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

            for label, pred in zip(targets, preds):
                total_per_cat[label] += 1
                if label == pred:
                    correct_per_cat[label] += 1

            if i == 0 and not sample_predictions:
                for j in range(min(5, len(targets))):
                    doc_idx = i * batch_size + j + len(docs) - len(X_test)
                    true_cat = labels[doc_idx]
                    pred_cat = learner.categories[preds[j].item()]
                    sample_predictions.append({
                        'text': docs[doc_idx][:100] + "...",
                        'true_category': true_cat,
                        'predicted_category': pred_cat
                    })

    overall_accuracy = correct / total
    per_cat_accuracy = correct_per_cat / total_per_cat.clamp(min=1)

    logger.info(f"\n=== Neural Network Test Results ===")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({correct}/{total})")
    logger.info("\nPer-Category Accuracy:")
    for i, cat in enumerate(learner.categories):
        logger.info(f"  {cat}: {per_cat_accuracy[i]:.4f} ({int(correct_per_cat[i])}/{int(total_per_cat[i])})")

    logger.info("\nSample Predictions:")
    for pred in sample_predictions:
        logger.info(f"  Text: {pred['text']}")
        logger.info(f"  True Category: {pred['true_category']}")
        logger.info(f"  Predicted Category: {pred['predicted_category']}")
        logger.info("  ---")

def main():
    kb = load_knowledge_base()
    if kb is None:
        return

    learner = LearningSystem(kb)
    X, y, num_classes = learner.prepare_data()
    if X is None:
        logger.error("No data available to test. Run the main script first.")
        return

    learner.load_model(input_size=X.shape[1], num_classes=num_classes)
    X_test, y_test, docs, labels = prepare_test_data(learner)
    if X_test is None:
        return

    test_model(learner, X_test, y_test, docs, labels)

if __name__ == "__main__":
    main()
