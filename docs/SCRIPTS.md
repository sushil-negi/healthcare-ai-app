# Scripts Documentation

## Overview
This document provides comprehensive documentation for all 30+ scripts in the healthcare-ai-app repository. Scripts are organized by category and include usage examples, parameters, and integration guidelines.

## Script Categories

### Data Management
Scripts for managing training data, validation, and quality checks.

#### `combine_all_datasets.py`
**Purpose**: Combine multiple healthcare training datasets into a single file.
```bash
python scripts/combine_all_datasets.py \
  --input-dir data/datasets/ \
  --output data/combined_healthcare_training.json \
  --format json
```
**Use Cases**: 
- Merging datasets from different sources
- Preparing training data for model training
- Creating comprehensive datasets for validation

#### `create_test_data.py`
**Purpose**: Generate test data for healthcare AI validation.
```bash
python scripts/create_test_data.py \
  --categories all \
  --samples-per-category 10 \
  --output data/test_healthcare_training.json
```
**Use Cases**:
- Creating test datasets for CI/CD
- Generating examples for new healthcare categories
- Building validation datasets

#### `healthcare_data_generator.py`
**Purpose**: Generate synthetic healthcare training data.
```bash
python scripts/healthcare_data_generator.py \
  --num-samples 1000 \
  --categories "adl_mobility,senior_medication" \
  --output data/generated_training.json
```
**Features**:
- Supports all 11 healthcare categories
- Includes medical disclaimers
- Crisis scenario generation

#### `data_quality_checks.py`
**Purpose**: Validate quality of healthcare training data.
```bash
python scripts/data_quality_checks.py \
  --input data/healthcare_training.json \
  --output reports/quality_report.json \
  --strict-mode
```
**Validation Checks**:
- JSON format validation
- Required fields verification
- Medical disclaimer presence
- Category distribution analysis

#### `validate_training_data.py`
**Purpose**: Comprehensive training data validation.
```bash
python scripts/validate_training_data.py \
  --input data/healthcare_training.json \
  --check-phi \
  --check-disclaimers \
  --output validation_report.json
```

### Model Training & Management
Scripts for training, evaluating, and managing healthcare AI models.

#### `create_mock_model.py`
**Purpose**: Create mock models for CI/CD testing.
```bash
python scripts/create_mock_model.py \
  --output models/healthcare_model.joblib \
  --include-crisis-detection
```
**Use Cases**:
- CI/CD pipeline testing
- Development environment setup
- Quick prototyping

#### `train_healthcare_model.py`
**Purpose**: Train the main healthcare AI model.
```bash
python scripts/train_healthcare_model.py \
  --input data/healthcare_training.json \
  --output models/healthcare_model.joblib \
  --test-split 0.2 \
  --cross-validate
```
**Parameters**:
- `--max-features`: TF-IDF vocabulary size (default: 5000)
- `--ngram-range`: N-gram range (default: 1,3)
- `--alpha`: Multinomial NB smoothing parameter

#### `train_real_healthcare_model.py`
**Purpose**: Train production-ready model with comprehensive validation.
```bash
python scripts/train_real_healthcare_model.py \
  --config training_config.yaml \
  --validate-crisis-detection \
  --output models/production_model.joblib
```

#### `enhanced_training_with_metrics.py`
**Purpose**: Advanced training with MLOps integration.
```bash
python scripts/enhanced_training_with_metrics.py \
  --experiment-name healthcare_training_v2 \
  --track-metrics \
  --log-artifacts
```

#### `healthcare_mock_model.py`
**Purpose**: Standalone mock model for testing.
```bash
python scripts/healthcare_mock_model.py \
  --serve \
  --port 8080 \
  --debug
```

### MLOps Integration
Scripts for integrating with the MLOps platform.

#### `submit_training_pipeline.py`
**Purpose**: Submit training jobs to MLOps platform.
```bash
python scripts/submit_training_pipeline.py \
  --pipeline healthcare_training_pipeline \
  --config pipelines/training_config.yaml \
  --wait-for-completion
```

#### `healthcare_pipeline_summary.py`
**Purpose**: Generate pipeline execution summaries.
```bash
python scripts/healthcare_pipeline_summary.py \
  --pipeline-id 12345 \
  --output reports/pipeline_summary.html
```

#### `model_interaction_guide.py`
**Purpose**: Guide for interacting with deployed models.
```bash
python scripts/model_interaction_guide.py \
  --model-endpoint http://localhost:8080 \
  --interactive
```

### Validation & Testing
Scripts for comprehensive system validation.

#### `hipaa_compliance_check.py`
**Purpose**: Validate HIPAA compliance across the system.
```bash
python scripts/hipaa_compliance_check.py \
  --check-data data/ \
  --check-code src/ \
  --output hipaa_compliance_report.json
```
**Checks**:
- PHI detection in code and data
- Audit logging verification
- Encryption validation
- Access control review

#### `comprehensive-pre-commit.py`
**Purpose**: Complete pre-commit validation suite.
```bash
python scripts/comprehensive-pre-commit.py \
  --fix-issues \
  --generate-report
```

#### `pre-commit-ci-validation.py`
**Purpose**: CI-specific pre-commit validation.
```bash
python scripts/pre-commit-ci-validation.py \
  --docker-builds \
  --port-checks \
  --security-scan
```

### Infrastructure & Deployment
Scripts for system deployment and infrastructure management.

#### `install-pre-commit-hook.sh`
**Purpose**: Install pre-commit hooks for development.
```bash
./scripts/install-pre-commit-hook.sh
```

#### `check-secrets.py`
**Purpose**: Scan for exposed secrets or credentials.
```bash
python scripts/check-secrets.py \
  --path . \
  --exclude node_modules \
  --fail-on-found
```

#### `healthcare_inference_wrapper.py`
**Purpose**: Wrapper for healthcare model inference.
```bash
python scripts/healthcare_inference_wrapper.py \
  --model models/healthcare_model.joblib \
  --query "What are balance exercises?" \
  --format json
```

### Monitoring & Analytics
Scripts for system monitoring and data analysis.

#### `data_drift_detector.py`
**Purpose**: Detect data drift in production queries.
```bash
python scripts/data_drift_detector.py \
  --baseline data/training_baseline.json \
  --current data/production_queries.json \
  --threshold 0.1
```

### Additional Utility Scripts
Other specialized scripts for specific tasks.

#### `setup_healthcare_env.py`
**Purpose**: Environment setup for healthcare AI development.
```bash
python scripts/setup_healthcare_env.py \
  --install-deps \
  --configure-env \
  --create-directories
```

## Usage Patterns

### Development Workflow
```bash
# 1. Setup environment
python scripts/setup_healthcare_env.py --install-deps

# 2. Generate test data
python scripts/create_test_data.py --categories all

# 3. Create mock model
python scripts/create_mock_model.py

# 4. Run validation
python scripts/comprehensive-pre-commit.py
```

### Training Workflow
```bash
# 1. Validate training data
python scripts/validate_training_data.py --input data/training.json

# 2. Check data quality
python scripts/data_quality_checks.py --input data/training.json

# 3. Train model
python scripts/train_healthcare_model.py --input data/training.json

# 4. Validate HIPAA compliance
python scripts/hipaa_compliance_check.py
```

### CI/CD Integration
```bash
# Pre-commit validation
python scripts/pre-commit-ci-validation.py

# Create test model
python scripts/create_mock_model.py

# Run comprehensive tests
python scripts/comprehensive-pre-commit.py --generate-report
```

## Configuration Files

Many scripts use configuration files:

### `training_config.yaml`
```yaml
model:
  type: sklearn_pipeline
  vectorizer:
    max_features: 5000
    ngram_range: [1, 3]
  classifier:
    alpha: 0.1

data:
  training_path: data/healthcare_training.json
  validation_split: 0.2

output:
  model_path: models/healthcare_model.joblib
  metrics_path: reports/training_metrics.json
```

### `validation_config.yaml`
```yaml
hipaa_compliance:
  check_phi: true
  check_encryption: true
  check_access_controls: true

data_quality:
  min_samples_per_category: 10
  required_disclaimer_rate: 0.9
  max_reading_level: 8

crisis_detection:
  test_scenarios: tests/crisis_scenarios.json
  min_recall: 0.95
```

## Best Practices

### Script Development
1. **Documentation**: Include docstrings and help text
2. **Error Handling**: Graceful failure with informative messages
3. **Logging**: Use consistent logging format
4. **Configuration**: Support configuration files
5. **Testing**: Include unit tests for complex scripts

### Usage Guidelines
1. **Environment**: Use virtual environments
2. **Dependencies**: Install from requirements.txt
3. **Permissions**: Check file permissions before execution
4. **Validation**: Always validate inputs
5. **Monitoring**: Log script execution for audit trails

### Integration
1. **CI/CD**: Use appropriate scripts in automation
2. **MLOps**: Integrate with platform workflows
3. **Monitoring**: Include in system monitoring
4. **Documentation**: Keep this document updated

## Troubleshooting

### Common Issues
1. **Module Import Errors**: Check PYTHONPATH and virtual environment
2. **File Not Found**: Verify file paths and permissions
3. **Configuration Errors**: Validate YAML/JSON configuration files
4. **Memory Issues**: Use streaming for large datasets
5. **Permission Denied**: Check file and directory permissions

### Support
For script-specific issues:
1. Check script help: `python script_name.py --help`
2. Review logs in `logs/` directory
3. Validate configuration files
4. Check system requirements
5. Review error messages and stack traces

## Maintenance

### Regular Updates
- Monthly review of script documentation
- Quarterly performance optimization
- Annual security review
- Continuous improvement based on usage patterns

### Version Control
- All scripts are version controlled
- Changes require code review
- Breaking changes documented in changelog
- Backward compatibility maintained when possible