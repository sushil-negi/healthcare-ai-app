# Healthcare AI Application - Crisis Detection & Response

## 🏥 **Enterprise Healthcare AI Platform**

A production-ready healthcare AI application designed for **crisis detection and mental health support**. This application integrates with the MLOps Platform to provide scalable, compliant, and reliable AI-powered healthcare services.

## Application Overview

The Healthcare AI application is an advanced healthcare chatbot that:
- **525K Training Dataset**: Uses real healthcare conversations for context-aware responses
- **Dynamic Response Generation**: Templates with personalization instead of static responses
- **11 Healthcare Categories**: ADL, mental health, senior care, respite care, disabilities
- **Crisis Detection**: Immediate detection and 988 hotline integration
- **Advanced AI Engine**: Knowledge-base mode with optional LLM integration
- **HIPAA Compliance**: Medical disclaimers and privacy protection
- **MLOps Integration**: Model management, monitoring, and continuous improvement

## Directory Structure

```
healthcare-ai-app/
├── src/models/          # Healthcare AI models and engines
├── data/               # Healthcare training datasets and templates
├── tests/              # Application-specific tests
├── web/                # Healthcare chat web interface
├── pipelines/          # Healthcare ML pipelines
├── scripts/            # Healthcare-specific scripts
├── monitoring/         # Healthcare application monitoring
└── deployment/         # Application deployment configurations
```

## Quick Start

### 1. Prerequisites
Ensure the MLOps platform is running:
```bash
cd ../platform/
docker-compose -f docker-compose.platform.yml up -d
```

### 2. Start Healthcare AI Application
```bash
cd healthcare-ai-app/
docker-compose -f docker-compose.app.yml up -d
```

### 3. Access Healthcare Chat
- **Web Interface**: http://localhost:8081
- **API Endpoint**: http://localhost:8080
- **Health Check**: http://localhost:8080/health

### 4. Test Healthcare Responses
```bash
# Crisis detection test
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to hurt myself"}'

# Balance exercises test
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are some balance exercises?"}'

# General healthcare query
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My elderly parent needs help with daily activities"}'
```

## Accessing Deployed System

### Docker Images
The Healthcare AI application is automatically built and published to GitHub Container Registry:

```bash
# Pull the latest images
docker pull ghcr.io/sushil-negi/healthcare-ai-app/healthcare-ai-healthcare-ai:latest
docker pull ghcr.io/sushil-negi/healthcare-ai-app/healthcare-ai-healthcare-web:latest
docker pull ghcr.io/sushil-negi/healthcare-ai-app/healthcare-ai-healthcare-metrics:latest
```

### Production Deployment
For production environments, use the published Docker images with appropriate environment configurations:

```bash
# Example production docker-compose override
docker compose -f docker-compose.app.yml \
  -f docker-compose.production.yml \
  up -d
```

### API Documentation
- **Development**: http://localhost:8080/docs (FastAPI auto-generated)
- **Note**: API docs are disabled in HIPAA compliance mode for security

## Healthcare AI Features

### Advanced AI Engine
- **525K Training Conversations**: Real healthcare data for context-aware responses
- **Dynamic Response Generation**: Templates with personalized context
- **Knowledge-Base Mode**: Sophisticated similarity matching from training data
- **Optional LLM Integration**: DialoGPT models for enhanced responses
- **Response Variation**: No more "canned" responses - each response is unique
- **Context Awareness**: Personalizes responses (e.g., "your mother" vs generic advice)

### Response Categories
1. **ADL (Activities of Daily Living)** - Mobility, transfers, daily tasks
2. **Senior Care** - Elderly care, aging in place, memory support
3. **Mental Health** - Anxiety, depression, emotional support
4. **Respite Care** - Caregiver support and relief services
5. **Disabilities** - Accessibility, adaptive equipment, inclusion
6. **Crisis Intervention** - Immediate crisis detection and resources
7. **Medication Management** - Pill organization, reminders, safety
8. **Mobility** - Walking aids, balance, physical therapy
9. **Nutrition** - Healthy eating, dietary guidance
10. **Social Support** - Community resources, support groups
11. **General Healthcare** - Medical information and guidance

### Crisis Detection System
- **Keywords**: Monitors for suicide, self-harm, and crisis language
- **Immediate Response**: Provides 988 Suicide & Crisis Lifeline
- **Safety Priority**: Crisis detection overrides all other responses
- **Professional Resources**: Directs to qualified mental health professionals

### Medical Compliance
- **Disclaimers**: All responses include medical disclaimers
- **Professional Guidance**: Recommends consulting healthcare providers
- **HIPAA Awareness**: Privacy-conscious response generation
- **Evidence-Based**: Responses based on healthcare best practices

## Application Services

| Service | Port | Purpose | Dependencies |
|---------|------|---------|--------------|
| Healthcare AI | 8080 | Core chatbot service | MLOps Platform |
| Healthcare Web | 8081 | Web chat interface | Healthcare AI |
| Healthcare Metrics | 8082 | Application metrics | Healthcare AI |

## Configuration

### Healthcare AI Settings
```yaml
healthcare_ai:
  model:
    version: "3.0.0"
    accuracy_threshold: 0.95
  crisis_detection:
    enabled: true
    hotline: "988"
  response_categories:
    - "adl_mobility"
    - "senior_care"
    - "mental_health"
    - "respite_care"
    - "disabilities"
    - "crisis"
    - "medication"
    - "mobility"
    - "nutrition"
    - "social"
    - "general"
```

### Environment Variables
```bash
# MLOps Platform Integration
MODEL_REGISTRY_URL=http://localhost:8000
EXPERIMENT_TRACKING_URL=http://localhost:8003
FEATURE_STORE_URL=http://localhost:8002

# Healthcare Configuration
CRISIS_DETECTION_ENABLED=true
MEDICAL_DISCLAIMER_REQUIRED=true
HIPAA_COMPLIANCE_MODE=true
```

## Model Training

### Training Data
- **Location**: `data/test_healthcare_training.json`
- **Format**: JSON with query, category, and response examples
- **Categories**: 11 healthcare categories with balanced samples
- **Quality**: Medical disclaimers in 85.7% of responses

### Training Pipeline
```bash
# Train healthcare model
python scripts/train_healthcare_model.py

# Validate training data
python scripts/validate_training_data.py

# Data quality checks
python scripts/data_quality_checks.py

# Real data collection
python scripts/real_data_collector.py
```

### Model Evaluation
- **Accuracy Target**: >95% classification accuracy
- **Crisis Detection**: >99% crisis detection rate
- **Response Quality**: Professional healthcare guidance
- **Compliance**: HIPAA-compliant data handling

## Testing

### Test Suites
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# End-to-end tests
HEALTHCARE_SERVICE_URL=http://localhost:8080 python -m pytest tests/e2e/ -v

# Crisis detection validation
python tests/crisis_detection_validation.py

# Response quality validation
python tests/response_quality_validation.py
```

### Test Coverage
- **Healthcare Logic**: Unit tests for response generation
- **Crisis Detection**: Comprehensive crisis scenario testing
- **ML Models**: Model accuracy and performance testing
- **API Integration**: End-to-end workflow validation
- **Compliance**: HIPAA and medical disclaimer verification

## Monitoring

### Healthcare Metrics
- **Response Quality Scores**: Empathy and professionalism ratings
- **Crisis Detection Rate**: Percentage of crisis situations detected
- **Category Classification**: Accuracy of healthcare category prediction
- **Medical Compliance**: Disclaimer presence and HIPAA compliance
- **User Interaction**: Response times and satisfaction metrics

### Application Monitoring
```bash
# Healthcare metrics endpoint
curl http://localhost:8082/metrics

# Application health
curl http://localhost:8080/health

# Response statistics
curl http://localhost:8080/stats
```

## Development

### Adding Healthcare Scenarios
1. Update training data in `data/test_healthcare_training.json`
2. Add response templates to knowledge base
3. Update category keywords for classification
4. Add test cases in `tests/`
5. Retrain model and validate

### Customizing Responses
1. Edit `src/models/healthcare-ai-k8s/src/healthcare_ai_engine.py`
2. Update response templates and keywords
3. Add medical disclaimers
4. Test crisis detection scenarios
5. Validate medical compliance

## Deployment

### Local Development
```bash
# Start with platform
docker-compose -f ../platform/docker-compose.platform.yml up -d
docker-compose -f docker-compose.app.yml up -d
```

### Kubernetes
```bash
# Deploy to healthcare-ai namespace
kubectl apply -f deployment/k8s/healthcare-ai-v2-deployment.yaml

# Check application status
kubectl get pods -n healthcare-ai-dev
```

### Production Deployment
- **Namespace**: `healthcare-ai-prod`
- **Replicas**: Multi-instance for high availability
- **Resources**: Adequate CPU/memory for ML inference
- **Secrets**: Secure configuration management
- **Monitoring**: Comprehensive health and performance monitoring

## Troubleshooting

### Common Issues
1. **Platform Connectivity**: Ensure MLOps platform services are accessible
2. **Model Loading**: Check model files and training data availability
3. **Crisis Detection**: Verify crisis keywords and response patterns
4. **Response Quality**: Monitor empathy scores and medical disclaimers

### Health Checks
```bash
# Application health
curl -f http://localhost:8080/health

# Crisis detection test
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am thinking about suicide"}'

# Category classification test
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are some balance exercises?"}'
```

## Documentation

- [Healthcare AI Features](docs/healthcare/features.md)
- [Crisis Detection Guide](docs/healthcare/crisis-detection.md)
- [Model Training Guide](docs/healthcare/model-training.md)
- [Medical Compliance](docs/healthcare/compliance.md)
- [API Reference](docs/API.md)
- [Scripts Documentation](docs/SCRIPTS.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Test Suite Summary](tests/TEST_SUITE_SUMMARY.md)