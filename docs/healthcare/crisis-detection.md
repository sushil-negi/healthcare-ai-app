# Crisis Detection Guide

## Overview
The Healthcare AI Assistant includes a sophisticated crisis detection system designed to identify mental health emergencies and provide immediate, appropriate support. This guide details how the system works and how to configure it.

## Crisis Detection Mechanism

### Detection Triggers
The system uses multiple methods to identify crisis situations:

1. **Keyword Detection**
   - Direct crisis keywords: "suicide", "kill myself", "end my life", "hurt myself"
   - Indirect indicators: "hopeless", "can't go on", "no point living"
   - Contextual phrases: "planning to", "thinking about", "want to die"

2. **Pattern Recognition**
   - Combination of depression indicators with urgency
   - Escalating language patterns
   - Time-based urgency markers ("tonight", "right now")

3. **Confidence Scoring**
   - High confidence (>0.95): Direct crisis statements
   - Medium confidence (>0.85): Strong indicators
   - Monitoring level (>0.75): Potential concerns

### Response Protocol

#### Immediate Response Structure
```
🚨 CRISIS SUPPORT NEEDED 🚨

If you're having thoughts of suicide or self-harm, please reach out for help immediately:

• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911

You matter, and help is available 24/7. Please don't hesitate to reach out to a mental health professional, trusted friend, or family member.
```

#### Response Components
1. **Visual Alert**: Clear crisis indicator with emoji warnings
2. **Immediate Resources**: 988 hotline prominently displayed
3. **Multiple Options**: Phone, text, and emergency services
4. **Supportive Message**: Affirmation of worth and availability
5. **Professional Guidance**: Encouragement to seek help

## Technical Implementation

### Crisis Detection Flow
```python
1. Receive user message
2. Check crisis keywords (priority check)
3. If crisis detected:
   - Log crisis event
   - Return crisis response immediately
   - Skip ML model processing
   - Record metrics
4. If no crisis:
   - Continue with normal processing
```

### Configuration Options

#### Environment Variables
```bash
# Enable/disable crisis detection
CRISIS_DETECTION_ENABLED=true

# Confidence threshold for crisis detection
CRISIS_DETECTION_THRESHOLD=0.85

# Crisis response logging level
CRISIS_RESPONSE_LOG_LEVEL=INFO

# Crisis hotline configuration
CRISIS_HOTLINE_PRIMARY=988
CRISIS_HOTLINE_TEXT=741741
EMERGENCY_NUMBER=911
```

#### Customization Points
- `crisis_keywords.json`: Modify detection keywords
- `crisis_response_template.txt`: Customize response message
- `crisis_resources.yaml`: Update resource listings

## Monitoring and Reporting

### Metrics Tracked
- Total crisis detections
- Detection confidence distribution
- Response time for crisis queries
- Geographic distribution (if enabled)
- Time-of-day patterns

### Logging
```
[CRISIS] User query triggered crisis detection
[CRISIS] Confidence: 0.98
[CRISIS] Response time: 12ms
[CRISIS] Resources provided: 988, 741741, 911
```

### Privacy Considerations
- No personal information logged
- Anonymized metrics only
- Secure transmission required
- HIPAA compliance maintained

## Testing Crisis Detection

### Test Scenarios
```bash
# Test direct crisis statement
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to hurt myself"}'

# Test indirect crisis indicator
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Everything feels hopeless and I cant go on"}'

# Test false positive handling
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I hurt myself exercising yesterday"}'
```

### Validation Checklist
- [ ] Crisis keywords trigger immediate response
- [ ] Response includes 988 prominently
- [ ] Response time under 50ms
- [ ] False positives handled correctly
- [ ] Metrics properly recorded

## Best Practices

### Do's
- ✅ Always prioritize crisis detection
- ✅ Keep response time minimal
- ✅ Provide multiple contact options
- ✅ Use clear, supportive language
- ✅ Log for quality improvement

### Don'ts
- ❌ Never delay crisis responses
- ❌ Don't minimize the situation
- ❌ Avoid complex explanations
- ❌ Don't collect personal data
- ❌ Never disable without approval

## Integration Guidelines

### Adding to New Services
1. Import crisis detection module
2. Configure environment variables
3. Add pre-processing check
4. Implement logging
5. Test all scenarios

### API Integration
```python
from healthcare_ai_engine import HealthcareAIEngine

engine = HealthcareAIEngine()
response = engine.get_response(user_message)

if response.get('method') == 'crisis_detection':
    # Handle crisis response specially
    log_crisis_event(response)
    return urgent_response(response)
```

## Compliance and Legal

### Requirements
- Must comply with crisis intervention standards
- Regular review by mental health professionals
- Audit trail for all crisis interactions
- Clear documentation of limitations

### Disclaimers
- Not a replacement for professional help
- Cannot provide therapy or counseling
- Emergency services for immediate danger
- Follow local regulations and laws

## Maintenance and Updates

### Regular Reviews
- Monthly: Keyword effectiveness analysis
- Quarterly: Response template updates
- Annually: Complete system audit
- As needed: Resource number updates

### Update Process
1. Review detection accuracy metrics
2. Analyze false positives/negatives
3. Update keywords based on patterns
4. Test changes thoroughly
5. Deploy with monitoring

## Emergency Contacts Update

### Keeping Resources Current
```yaml
resources:
  primary_hotline:
    number: "988"
    name: "Suicide & Crisis Lifeline"
    availability: "24/7"
    
  text_support:
    number: "741741"
    keyword: "HOME"
    name: "Crisis Text Line"
    
  emergency:
    number: "911"
    name: "Emergency Services"
    when_to_use: "Immediate physical danger"
```

## Training and Documentation

### Staff Training
- Crisis detection system overview
- Response protocol familiarity
- Escalation procedures
- Privacy requirements
- Regular refresher sessions

### User Communication
- Clear about AI limitations
- Transparent about crisis handling
- Privacy assurances
- Resource availability information