# Demo LLM Training Datasets

## Quick Start Datasets (Built-in)

### 1. **Simple Text Patterns** (Easiest)
Perfect for quick demos and testing the pipeline.

```python
# Simple repetitive patterns
patterns = [
    "The weather today is sunny and warm.",
    "The weather today is cloudy and cool.",
    "The weather today is rainy and cold.",
    "Machine learning is transforming technology.",
    "Machine learning is revolutionizing industries.",
    "Machine learning is changing the world.",
]
```

### 2. **Technical Documentation** (Domain-specific)
Good for demonstrating domain adaptation.

```python
tech_docs = [
    "To install Python, run: pip install python",
    "To create a virtual environment, use: python -m venv myenv",
    "To activate the environment, run: source myenv/bin/activate",
    "Docker containers provide isolated environments for applications.",
    "Kubernetes orchestrates containerized applications at scale.",
    "MLflow tracks experiments and manages model lifecycle.",
]
```

### 3. **Customer Support Responses** (Business use case)
Demonstrates practical applications.

```python
support_responses = [
    "Thank you for contacting support. How can I help you today?",
    "I understand your concern. Let me look into that for you.",
    "I've escalated your issue to our technical team.",
    "Is there anything else I can assist you with?",
    "Your satisfaction is our top priority.",
]
```

## Medium-Complexity Datasets

### 4. **News Headlines Dataset**
- **Size**: 10K-100K headlines
- **Source**: Can generate from news APIs or use public datasets
- **Good for**: Learning concise, informative text generation

### 5. **Product Descriptions**
- **Size**: 5K-50K descriptions
- **Source**: E-commerce datasets
- **Good for**: Learning descriptive and persuasive text

### 6. **Code Comments Dataset**
- **Size**: 10K-100K code-comment pairs
- **Source**: GitHub repositories
- **Good for**: Technical documentation generation

## Publicly Available Datasets

### 7. **WikiText-103** (Recommended for demos)
- **Size**: 103M tokens
- **Download**: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
- **Good for**: General language understanding

```bash
# Download WikiText-103
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
```

### 8. **TinyStories** (Lightweight narrative)
- **Size**: 2.1M short stories
- **Source**: https://huggingface.co/datasets/roneneldan/TinyStories
- **Good for**: Creative text generation

### 9. **BookCorpus** (Subset)
- **Size**: Variable (can use subset)
- **Source**: https://huggingface.co/datasets/bookcorpus
- **Good for**: Long-form text understanding

### 10. **OpenWebText** (Web content)
- **Size**: 40GB (can use small subset)
- **Source**: https://huggingface.co/datasets/openwebtext
- **Good for**: Diverse internet text

## Custom Dataset Creation

### Option A: Domain-Specific Corpus
Create a focused dataset for your use case:

```python
# Financial news dataset
financial_texts = [
    "Stock markets rallied today as investors showed renewed confidence.",
    "The Federal Reserve announced interest rates will remain unchanged.",
    "Tech stocks led the gains with a 3% increase across the sector.",
    # ... add more examples
]

# Medical/Healthcare dataset  
medical_texts = [
    "Patient presented with symptoms including fever and fatigue.",
    "Treatment plan includes antibiotics and rest.",
    "Follow-up appointment scheduled in two weeks.",
    # ... add more examples
]

# Legal document dataset
legal_texts = [
    "This agreement is entered into between Party A and Party B.",
    "The terms and conditions outlined herein are legally binding.",
    "Both parties agree to the stipulations set forth in this document.",
    # ... add more examples
]
```

### Option B: Synthetic Data Generation
Generate training data programmatically:

```python
import random

# Template-based generation
templates = [
    "The {adjective} {noun} {verb} {adverb}.",
    "In {location}, {subject} {action} {object}.",
    "{Person} said that {statement}.",
]

adjectives = ["quick", "lazy", "beautiful", "intelligent"]
nouns = ["dog", "cat", "developer", "algorithm"]
verbs = ["runs", "jumps", "codes", "processes"]
adverbs = ["quickly", "slowly", "efficiently", "carefully"]

# Generate 1000 sentences
synthetic_data = []
for _ in range(1000):
    template = random.choice(templates)
    # Fill in the template...
```

## Quick Implementation

Here's how to use these datasets with your model:

```python
# Save this as prepare_training_data.py
import json
import os

def prepare_simple_dataset():
    """Prepare a simple dataset for quick testing"""
    
    # Simple patterns for demo
    training_texts = [
        "The future of AI is bright and full of possibilities.",
        "Machine learning transforms how we solve complex problems.",
        "Deep learning models can understand patterns in data.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to see and interpret images.",
        "Reinforcement learning teaches agents through trial and error.",
        "Transfer learning allows models to apply knowledge to new tasks.",
        "Federated learning enables privacy-preserving model training.",
        "Quantum computing will revolutionize machine learning.",
        "Edge AI brings intelligence to IoT devices.",
    ] * 100  # Repeat for more data
    
    # Add variations
    for i in range(len(training_texts)):
        if i % 2 == 0:
            training_texts[i] = training_texts[i].lower()
        if i % 3 == 0:
            training_texts[i] = training_texts[i] + " This is amazing!"
    
    # Save to file
    os.makedirs('/app/artifacts/data', exist_ok=True)
    with open('/app/artifacts/data/training_data.json', 'w') as f:
        json.dump({
            'texts': training_texts,
            'metadata': {
                'dataset': 'simple_demo',
                'size': len(training_texts),
                'domain': 'AI/ML'
            }
        }, f, indent=2)
    
    print(f"Created training dataset with {len(training_texts)} examples")

if __name__ == "__main__":
    prepare_simple_dataset()
```

## Dataset Recommendations by Use Case

1. **For Quick Demo** (5-10 minutes training):
   - Use the simple text patterns (100-1000 examples)
   - Focus on showing the pipeline, not model quality

2. **For Realistic Demo** (30-60 minutes training):
   - Use WikiText-103 subset (1-10MB)
   - Shows actual learning progress

3. **For Domain-Specific Demo**:
   - Create custom dataset in your domain
   - 5,000-10,000 examples minimum

4. **For Production Testing**:
   - Use full datasets like OpenWebText
   - Requires GPU and longer training time

## Loading Data into the Pipeline

To use any of these datasets:

1. Place the data in `/data` directory
2. Update the training script to load your dataset
3. Run the pipeline: `./scripts/docker-pipeline.sh train`

The model will automatically process and train on your data!