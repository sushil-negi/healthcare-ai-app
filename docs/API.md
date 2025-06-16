# Healthcare AI API Documentation

## Base URL
- **Development**: `http://localhost:8080`
- **Production**: Configure based on your deployment

## Authentication
Currently, the API does not require authentication. For production use, implement appropriate authentication mechanisms.

## Endpoints

### Chat Endpoint
**POST** `/chat`

Send a healthcare-related query and receive an AI-generated response.

#### Request
```json
{
  "message": "string"
}
```

#### Response
```json
{
  "response": "string",
  "category": "string",
  "confidence": 0.0-1.0,
  "method": "ml_model|crisis_detection|fallback",
  "timing": {
    "crisis_check_ms": 0.0,
    "ml_inference_ms": 0.0,
    "total_ms": 0.0
  }
}
```

#### Example
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What exercises can help with balance?"}'
```

### Health Check
**GET** `/health`

Returns the health status of the service.

#### Response
```json
{
  "status": "healthy",
  "service": "Healthcare AI Assistant",
  "version": "3.0.0",
  "timestamp": "2025-06-16T17:25:50.903129",
  "engine_stats": {
    "model_loaded": true,
    "categories": 11,
    "category_list": ["adl_mobility", "..."],
    "total_responses": 0,
    "cache_size": 0,
    "conversation_history": 0,
    "model_type": "TfidfVectorizer + MultinomialNB"
  }
}
```

### Statistics
**GET** `/stats`

Returns usage statistics for the service.

#### Response
```json
{
  "total_requests": 0,
  "crisis_detections": 0,
  "ml_predictions": 0,
  "fallback_responses": 0,
  "average_response_time_ms": 0.0,
  "categories": {
    "adl_mobility": 0,
    "adl_self_care": 0,
    "senior_medication": 0,
    "...": 0
  }
}
```

### Metrics
**GET** `/metrics`

Returns Prometheus-formatted metrics for monitoring.

### Ready Check
**GET** `/ready`

Returns readiness status for Kubernetes probes.

## Response Categories

The AI categorizes responses into the following healthcare categories:

- `adl_mobility` - Mobility and movement assistance
- `adl_self_care` - Daily self-care activities
- `senior_medication` - Medication management for seniors
- `senior_social` - Social activities and engagement
- `mental_health_anxiety` - Anxiety support
- `mental_health_depression` - Depression support
- `crisis_mental_health` - Crisis intervention (triggers 988 resources)
- `caregiver_respite` - Respite care for caregivers
- `caregiver_burnout` - Caregiver burnout support
- `disability_equipment` - Disability equipment and aids
- `disability_rights` - Disability rights and advocacy
- `general_healthcare` - General healthcare queries

## Crisis Detection

The system automatically detects crisis situations and provides immediate resources:

- Triggers on keywords indicating self-harm, suicide, or immediate danger
- Returns 988 Suicide & Crisis Lifeline information
- Overrides normal ML response generation
- Logged separately for monitoring

## Error Handling

The API returns standard HTTP status codes:

- `200 OK` - Successful response
- `400 Bad Request` - Invalid request format
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

Error responses include details:
```json
{
  "detail": "Error description"
}
```

## Rate Limiting

Currently not implemented. For production use, implement appropriate rate limiting.

## HIPAA Compliance

When `HIPAA_COMPLIANCE_MODE=true`:
- API documentation endpoints are disabled
- Additional security headers are enforced
- PHI anonymization is enabled
- Audit logging is enhanced