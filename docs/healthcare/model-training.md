# Model Training Guide

## Overview
This guide provides comprehensive instructions for training the Healthcare AI model, from data preparation through deployment. The model uses a combination of TF-IDF vectorization and Multinomial Naive Bayes classification to categorize healthcare queries.

## Prerequisites

### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries
- scikit-learn>=1.1.0
- numpy>=1.23.0
- pandas>=1.5.0
- joblib>=1.3.0
- mlflow>=2.0.0 (for experiment tracking)

## Training Data Preparation

### Data Format
Training data must be in JSON format with the following structure:
```json
[
  {
    "query": "User's healthcare question",
    "category": "healthcare_category",
    "response": "AI assistant's response"
  }
]
```

### Category Labels
The model recognizes 11 healthcare categories:
- `adl_mobility` - Mobility assistance
- `adl_self_care` - Daily self-care activities
- `senior_medication` - Medication management
- `senior_social` - Social activities for seniors
- `mental_health_anxiety` - Anxiety support
- `mental_health_depression` - Depression support
- `caregiver_respite` - Respite care
- `caregiver_burnout` - Caregiver burnout
- `disability_equipment` - Disability equipment
- `disability_rights` - Disability rights
- `crisis_mental_health` - Crisis intervention

### Data Validation
Run the validation script before training:
```bash
python scripts/validate_training_data.py --input data/healthcare_training.json
```

Validation checks:
- JSON format validity
- Required fields presence
- Category label validity
- Response quality metrics
- Crisis response verification

## Training Process

### 1. Local Training
For quick local training with default parameters:
```bash
python scripts/train_healthcare_model.py \
  --input data/healthcare_training.json \
  --output models/healthcare_model.joblib
```

### 2. Advanced Training Options
```bash
python scripts/train_healthcare_model.py \
  --input data/healthcare_training.json \
  --output models/healthcare_model.joblib \
  --test-split 0.2 \
  --random-state 42 \
  --max-features 5000 \
  --ngram-range 1,3 \
  --min-df 2 \
  --use-idf true
```

Parameters:
- `--test-split`: Proportion of data for testing (default: 0.2)
- `--random-state`: Random seed for reproducibility
- `--max-features`: Maximum vocabulary size
- `--ngram-range`: N-gram range for text features
- `--min-df`: Minimum document frequency
- `--use-idf`: Enable TF-IDF weighting

### 3. MLOps Platform Training
For production training with experiment tracking:
```bash
python scripts/submit_training_pipeline.py \
  --pipeline healthcare_training_pipeline \
  --config pipelines/training_config.yaml
```

Training configuration (training_config.yaml):
```yaml
pipeline:
  name: healthcare_model_training
  version: 1.0.0

data:
  training_path: data/healthcare_training.json
  validation_split: 0.2
  stratify: true

model:
  type: sklearn_pipeline
  vectorizer:
    type: TfidfVectorizer
    max_features: 5000
    ngram_range: [1, 3]
    min_df: 2
    use_idf: true
  classifier:
    type: MultinomialNB
    alpha: 0.1

training:
  cross_validation: true
  cv_folds: 5
  metrics:
    - accuracy
    - f1_macro
    - precision_macro
    - recall_macro

output:
  model_registry: true
  artifact_path: models/
  register_name: healthcare_ai_model
```

## Model Evaluation

### Performance Metrics
After training, evaluate the model:
```bash
python scripts/evaluate_healthcare_model.py \
  --model models/healthcare_model.joblib \
  --test-data data/test_healthcare.json
```

Key metrics to monitor:
- **Overall Accuracy**: >85% expected
- **Per-Category F1 Score**: >0.80 for each category
- **Crisis Detection Recall**: Must be >0.95
- **False Positive Rate**: <5% for crisis detection

### Validation Tests
Run comprehensive validation:
```bash
# Healthcare-specific validation
python tests/healthcare_model_validation.py

# Crisis detection validation
python tests/crisis_detection_validation.py

# Response quality validation
python tests/response_quality_validation.py
```

## Model Optimization

### Hyperparameter Tuning
Use grid search for optimization:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'vectorizer__max_features': [3000, 5000, 10000],
    'vectorizer__ngram_range': [(1,2), (1,3), (2,3)],
    'classifier__alpha': [0.01, 0.1, 1.0]
}

# Run grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)
```

### Feature Engineering
Improve model performance:
1. **Add domain-specific features**:
   - Medical term frequency
   - Urgency indicators
   - Question type classification

2. **Text preprocessing**:
   - Remove medical abbreviations
   - Normalize symptom descriptions
   - Handle typos and variations

3. **Category balancing**:
   - Use SMOTE for imbalanced categories
   - Adjust class weights
   - Augment underrepresented categories

## Deployment

### 1. Model Export
```bash
# Export trained model
python scripts/export_model.py \
  --model models/healthcare_model.joblib \
  --format joblib \
  --compress true
```

### 2. Model Versioning
```bash
# Register with MLflow
mlflow models register \
  --model-uri runs:/<run-id>/model \
  --name healthcare_ai_model \
  --tags version=1.0.0,stage=production
```

### 3. Service Integration
Update the service to use new model:
```python
# In healthcare_ai_engine.py
MODEL_PATH = "models/healthcare_model_v1.0.0.joblib"
```

### 4. A/B Testing
For gradual rollout:
```yaml
model_config:
  stable:
    path: models/healthcare_model_v0.9.0.joblib
    traffic: 0.8
  canary:
    path: models/healthcare_model_v1.0.0.joblib
    traffic: 0.2
```

## Monitoring and Maintenance

### Performance Monitoring
Track key metrics in production:
- Response accuracy by category
- Inference latency
- Model confidence distribution
- Drift detection metrics

### Regular Retraining
Schedule based on:
- Performance degradation (accuracy <80%)
- New healthcare guidelines
- Vocabulary drift
- User feedback patterns

### Retraining Pipeline
```bash
# Automated retraining
python scripts/automated_retraining.py \
  --performance-threshold 0.80 \
  --data-source production_feedback \
  --schedule weekly
```

## Best Practices

### Data Quality
1. **Diverse Examples**: Include variations for each category
2. **Medical Accuracy**: Validate with healthcare professionals
3. **Crisis Coverage**: Comprehensive crisis scenarios
4. **Regular Updates**: Incorporate new medical guidelines

### Model Safety
1. **Conservative Thresholds**: Higher confidence for medical advice
2. **Fallback Mechanisms**: Default to safe responses
3. **Human Review**: Regular expert validation
4. **Audit Trail**: Log all predictions

### Compliance
1. **HIPAA Compliance**: No PII in training data
2. **Medical Disclaimers**: Include in training responses
3. **Bias Testing**: Regular fairness assessments
4. **Documentation**: Maintain training records

## Troubleshooting

### Common Issues

1. **Low Accuracy**
   - Check data quality and balance
   - Increase training data size
   - Adjust hyperparameters
   - Review feature engineering

2. **Crisis Detection Failures**
   - Expand crisis keyword list
   - Lower detection threshold
   - Add more crisis examples
   - Review false negatives

3. **Category Confusion**
   - Analyze confusion matrix
   - Add distinguishing features
   - Clarify category boundaries
   - Increase training examples

4. **Model Size Issues**
   - Reduce vocabulary size
   - Use feature selection
   - Compress model file
   - Consider model quantization

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Healthcare NLP Best Practices](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7233093/)
- [Crisis Intervention Guidelines](https://suicidepreventionlifeline.org/)