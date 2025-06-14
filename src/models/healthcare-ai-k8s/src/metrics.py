"""
Prometheus metrics for Healthcare AI service
"""

import time
from functools import wraps

from prometheus_client import Counter, Gauge, Histogram, Summary

# Request metrics
request_count = Counter(
    "healthcare_ai_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

request_duration = Histogram(
    "healthcare_ai_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
)

# Model performance metrics
model_accuracy = Gauge(
    "healthcare_ai_model_accuracy", "Current model accuracy", ["model_version"]
)

crisis_detection_rate = Gauge(
    "healthcare_ai_crisis_detection_rate",
    "Rate of successful crisis detections",
    ["model_version"],
)

empathy_score = Gauge(
    "healthcare_ai_empathy_score",
    "Average empathy score in responses",
    ["model_version"],
)

# Category metrics
category_requests = Counter(
    "healthcare_ai_category_requests_total", "Total requests per category", ["category"]
)

# Response quality metrics
response_clarity = Histogram(
    "healthcare_ai_response_clarity",
    "Response clarity score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

response_completeness = Histogram(
    "healthcare_ai_response_completeness",
    "Response completeness score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Active users
active_users = Gauge(
    "healthcare_ai_active_users", "Number of active users in the last 5 minutes"
)

# Model metadata
model_info = Gauge(
    "healthcare_ai_model_info", "Model information", ["version", "stage", "name"]
)


def track_request(method: str, endpoint: str):
    """Decorator to track request metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                request_count.labels(
                    method=method, endpoint=endpoint, status=status
                ).inc()
                request_duration.labels(method=method, endpoint=endpoint).observe(
                    duration
                )

        return wrapper

    return decorator


def update_model_metrics(
    accuracy: float, crisis_rate: float, empathy: float, version: str = "unknown"
):
    """Update model performance metrics"""
    model_accuracy.labels(model_version=version).set(accuracy)
    crisis_detection_rate.labels(model_version=version).set(crisis_rate)
    empathy_score.labels(model_version=version).set(empathy)


def track_category_request(category: str):
    """Track requests by category"""
    category_requests.labels(category=category).inc()


def update_response_quality(clarity: float, completeness: float):
    """Update response quality metrics"""
    response_clarity.observe(clarity)
    response_completeness.observe(completeness)


def set_active_users(count: int):
    """Set active users count"""
    active_users.set(count)


def set_model_info(version: str, stage: str, name: str):
    """Set model information"""
    model_info.labels(version=version, stage=stage, name=name).set(1)
