# Autonomous AI Researcher System

A self-improving AI research assistant with secure code execution, distributed processing, and formal verification capabilities.

## Features

- 🔒 Security sandboxing for generated code
- 📊 Vector database integration (ChromaDB)
- ⚡ Distributed task processing (Celery/Redis)
- ✅ Formal verification of algorithms (Z3)
- 🤖 Large language model integration (GPT-Neo 2.7B)
- 📚 Automated literature review (arXiv)

## Installation

### Requirements
- Python 3.8+
- Redis server
- 16GB+ RAM (recommended)
- 10GB+ disk space for models

### 1. Core Dependencies
```bash
pip install -r requirements.txt

```
2. Additional Setup

# spaCy language model
```py
python -m spacy download en_core_web_lg```

# Redis server (Ubuntu)
```sudo apt-get install redis-server```

Getting Started
1. Start Services
# Start Redis in separate terminal
```redis-server```

```
# Start Celery worker in another terminal
celery -A ai_researcher worker --loglevel=info```

2. Run the System
```py
from research_framework import DistributedResearchFramework
researcher = DistributedResearchFramework()
researcher.distributed_research(["AI safety", "quantum computing"])
```
Configuration
Environment Setup
# For GPU acceleration (CUDA)
```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```
Alternative Conda Setup
```bash
conda create -n researcher python=3.9
conda activate researcher
conda install -c conda-forge redis
pip install -r requirements.txt
```
Verification
Test system functionality:

```py
import spacy
from transformers import pipeline
from chromadb import Client
```
# Check core components
```py
print("SpaCy version:", spacy.__version__)
print("Transformers test:", pipeline('text-generation')("Hello")[0]['generated_text'])
print("ChromaDB collections:", Client().list_collections())
```
Important Notes
Model Downloads:

GPT-Neo 2.7B (~10GB) auto-downloads on first run

Sentence Transformer caches at ~/.cache/torch/sentence_transformers

Port Requirements:

Redis: 6379 (default)

Celery: 6379 (broker)

Security:

Never run untrusted code without sandbox

Default configuration uses RestrictedPython

License
MIT License - see LICENSE for details



