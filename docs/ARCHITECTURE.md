# Healthcare AI Application Architecture

## System Overview
The Healthcare AI Application is a microservices-based system designed to provide HIPAA-compliant healthcare guidance and crisis detection. It integrates with the MLOps platform for model management and monitoring.

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Healthcare    │    │   Healthcare    │    │   Healthcare    │
│   Web Interface │    │   AI Service    │    │   Metrics       │
│   Port: 8081    │    │   Port: 8080    │    │   Port: 8085    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    │   Cache Store   │
                    │   Port: 6379    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   MLOps         │
                    │   Platform      │
                    │   Integration   │
                    └─────────────────┘
```

## Core Components

### 1. Healthcare AI Service (Port 8080)
**Location**: `src/models/healthcare-ai/`
**Purpose**: Core AI processing service

#### Key Features:
- ML-based query classification
- Crisis detection with immediate response
- Medical disclaimer integration
- HIPAA compliance enforcement

#### Endpoints:
- `POST /chat` - Process healthcare queries
- `GET /health` - Service health check
- `GET /stats` - Usage statistics
- `GET /metrics` - Prometheus metrics

#### Technologies:
- FastAPI web framework
- Scikit-learn ML models
- TF-IDF vectorization
- Multinomial Naive Bayes classification

### 2. Healthcare Web Interface (Port 8081)
**Location**: `web/`
**Purpose**: User-friendly chat interface

#### Features:
- Responsive web design
- Real-time chat interface
- Crisis warning system
- Session management

#### Technologies:
- HTML5/CSS3/JavaScript
- Bootstrap for responsive design
- WebSocket for real-time updates
- Progressive Web App (PWA) features

### 3. Healthcare Metrics (Port 8085)
**Location**: `monitoring/`
**Purpose**: Prometheus metrics collection

#### Metrics Collected:
- Request volume and latency
- Crisis detection counts
- Model accuracy metrics
- System health indicators

## Data Flow

### Standard Query Processing
```
User Query → Web Interface → AI Service → ML Model → Response
     ↓              ↓            ↓          ↓          ↓
  Session      API Proxy    Validation  Inference  Disclaimer
   Data         Logging      Checks      Engine     Addition
```

### Crisis Detection Flow
```
Crisis Query → Keyword Detection → Immediate Response
      ↓              ↓                    ↓
   Logging      Crisis Protocol     988 Resources
  (Anonymous)    (Skip ML Model)    (Emergency Info)
```

## Healthcare Categories

The system handles 11 distinct healthcare categories:

1. **ADL (Activities of Daily Living)**
   - `adl_mobility` - Movement assistance
   - `adl_self_care` - Daily care activities

2. **Senior Care**
   - `senior_medication` - Medication management
   - `senior_social` - Social engagement

3. **Mental Health**
   - `mental_health_anxiety` - Anxiety support
   - `mental_health_depression` - Depression support
   - `crisis_mental_health` - Crisis intervention

4. **Caregiver Support**
   - `caregiver_respite` - Respite care
   - `caregiver_burnout` - Burnout prevention

5. **Disability Support**
   - `disability_equipment` - Adaptive equipment
   - `disability_rights` - Rights and advocacy

## Security & Compliance

### HIPAA Compliance
- No PHI collection or storage
- Anonymous session handling
- Encrypted data transmission
- Audit logging for all interactions

### Security Measures
- Input validation and sanitization
- Rate limiting (configurable)
- Secure headers enforcement
- Regular security scanning

## Integration Points

### MLOps Platform Integration
```yaml
Platform Services:
  - Model Registry: http://localhost:8000
  - Experiment Tracking: http://localhost:8003
  - Feature Store: http://localhost:8002
  - Pipeline Orchestrator: http://localhost:8004
```

### External Services
- 988 Suicide & Crisis Lifeline (crisis detection)
- Crisis Text Line (text support)
- Emergency Services (911 integration)

## Deployment Architecture

### Local Development
```
Docker Compose
├── healthcare-ai (AI Service)
├── healthcare-web (Web Interface)
├── healthcare-metrics (Monitoring)
└── redis (Caching)
```

### Production Deployment
```
Kubernetes Cluster
├── healthcare-ai-deployment
├── healthcare-web-deployment  
├── healthcare-metrics-deployment
├── redis-deployment
└── ingress-controller
```

## Configuration Management

### Environment Variables
```bash
# Core Configuration
HEALTHCARE_AI_URL=http://healthcare-ai:8080
PORT=8081

# Feature Flags
CRISIS_DETECTION_ENABLED=true
HIPAA_COMPLIANCE_MODE=true
MEDICAL_DISCLAIMER_REQUIRED=true

# Performance Settings
MAX_CONCURRENT_REQUESTS=100
RESPONSE_CACHE_TTL_SECONDS=300
MODEL_WARMUP_ENABLED=true
```

### Configuration Files
- `docker-compose.app.yml` - Local development
- `docker-compose.healthcare-ci.yml` - CI/CD testing
- `deployment/k8s/` - Kubernetes manifests
- `pipelines/training_config.yaml` - ML training

## Monitoring & Observability

### Health Checks
- Service health endpoints
- Kubernetes readiness probes
- Liveness probes for auto-restart

### Metrics Collection
- Prometheus metrics export
- Application performance monitoring
- Business metrics tracking

### Logging
- Structured JSON logging
- Centralized log aggregation
- Privacy-compliant audit trails

## Scalability Considerations

### Horizontal Scaling
- Stateless service design
- Load balancer compatibility
- Redis session sharing

### Performance Optimization
- Model inference caching
- Connection pooling
- Async request processing

## Development Workflow

### Local Setup
1. Start MLOps platform dependencies
2. Launch healthcare services
3. Configure environment variables
4. Run validation tests

### CI/CD Pipeline
1. Code quality checks
2. Security scanning
3. Docker image building
4. Automated testing
5. Deployment automation

## Future Enhancements

### Planned Features
- Multi-language support
- Voice interface integration
- Advanced personalization
- Enhanced crisis detection

### Technical Improvements
- GraphQL API implementation
- Advanced caching strategies
- Machine learning model updates
- Enhanced monitoring dashboards