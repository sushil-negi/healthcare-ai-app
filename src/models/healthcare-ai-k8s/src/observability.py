"""
Observability and monitoring setup for Healthcare AI
Includes structured logging, distributed tracing, and metrics
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram


# Configure structured logging
def configure_logging():
    """Configure structured logging with JSON output"""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        handlers=[logging.StreamHandler()],
    )


def setup_tracing():
    """Setup distributed tracing with Jaeger"""

    jaeger_host = os.getenv("JAEGER_HOST", "localhost")
    jaeger_port = int(os.getenv("JAEGER_PORT", "14268"))

    # Configure tracer provider
    trace.set_tracer_provider(TracerProvider())

    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )

    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    # Auto-instrument libraries
    RequestsInstrumentor().instrument()
    RedisInstrumentor().instrument()

    return trace.get_tracer(__name__)


# Prometheus metrics
healthcare_requests_total = Counter(
    "healthcare_ai_requests_total",
    "Total healthcare AI requests",
    ["category", "method", "status"],
)

healthcare_response_time = Histogram(
    "healthcare_ai_response_time_seconds",
    "Healthcare AI response time",
    ["category", "method"],
)

healthcare_confidence_score = Histogram(
    "healthcare_ai_confidence_score", "Healthcare AI confidence score", ["category"]
)

crisis_responses_total = Counter(
    "healthcare_ai_crisis_responses_total", "Total crisis responses generated"
)

model_accuracy_gauge = Gauge("healthcare_ai_model_accuracy", "Current model accuracy")

active_sessions_gauge = Gauge(
    "healthcare_ai_active_sessions", "Number of active user sessions"
)


class HealthcareAILogger:
    """Structured logger for Healthcare AI with privacy protection"""

    def __init__(self, service_name: str = "healthcare-ai"):
        self.service_name = service_name
        self.logger = structlog.get_logger(service=service_name)

    def log_request(
        self,
        category: str,
        confidence: float,
        response_time: float,
        method: str = "unknown",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ):
        """Log healthcare AI request with privacy protection"""

        # Remove any potentially sensitive information
        safe_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["user_query", "user_input", "response_text"]
        }

        self.logger.info(
            "healthcare_request_processed",
            category=category,
            confidence=confidence,
            response_time=response_time,
            method=method,
            user_id=user_id and f"user_{hash(user_id) % 10000}",  # Anonymize
            session_id=session_id
            and f"session_{hash(session_id) % 10000}",  # Anonymize
            timestamp=datetime.utcnow().isoformat(),
            **safe_kwargs,
        )

        # Update Prometheus metrics
        healthcare_requests_total.labels(
            category=category, method=method, status="success"
        ).inc()

        healthcare_response_time.labels(category=category, method=method).observe(
            response_time
        )

        healthcare_confidence_score.labels(category=category).observe(confidence)

        # Special handling for crisis responses
        if category == "crisis":
            crisis_responses_total.inc()
            self.logger.warning(
                "crisis_response_generated",
                category=category,
                confidence=confidence,
                timestamp=datetime.utcnow().isoformat(),
                alert_type="crisis_response",
                priority="high",
            )

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        category: str = "unknown",
        **kwargs,
    ):
        """Log errors with context"""

        self.logger.error(
            "healthcare_ai_error",
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            context=context or {},
            timestamp=datetime.utcnow().isoformat(),
            **kwargs,
        )

        # Update error metrics
        healthcare_requests_total.labels(
            category=category, method="unknown", status="error"
        ).inc()

    def log_model_training(
        self,
        accuracy: float,
        loss: float,
        training_time: float,
        data_size: int,
        **kwargs,
    ):
        """Log model training metrics"""

        self.logger.info(
            "model_training_completed",
            accuracy=accuracy,
            loss=loss,
            training_time=training_time,
            data_size=data_size,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs,
        )

        # Update model accuracy gauge
        model_accuracy_gauge.set(accuracy)

        # Alert on low accuracy
        if accuracy < 0.90:
            self.logger.warning(
                "low_model_accuracy",
                accuracy=accuracy,
                threshold=0.90,
                alert_type="low_accuracy",
                priority="high",
                timestamp=datetime.utcnow().isoformat(),
            )

    def log_performance_issue(
        self,
        response_time: float,
        threshold: float = 1.0,
        category: str = "unknown",
        **kwargs,
    ):
        """Log performance issues"""

        if response_time > threshold:
            self.logger.warning(
                "slow_response_detected",
                response_time=response_time,
                threshold=threshold,
                category=category,
                alert_type="slow_response",
                priority="medium",
                timestamp=datetime.utcnow().isoformat(),
                **kwargs,
            )


def trace_healthcare_function(operation_name: str):
    """Decorator for tracing healthcare AI functions"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)

            with tracer.start_as_current_span(operation_name) as span:
                # Add healthcare-specific attributes
                span.set_attribute("service.name", "healthcare-ai")
                span.set_attribute("operation.name", operation_name)

                # Extract healthcare context if available
                if "category" in kwargs:
                    span.set_attribute("healthcare.category", kwargs["category"])

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    # Add result attributes
                    if isinstance(result, dict):
                        if "category" in result:
                            span.set_attribute(
                                "healthcare.category", result["category"]
                            )
                        if "confidence" in result:
                            span.set_attribute(
                                "healthcare.confidence", result["confidence"]
                            )

                    span.set_attribute("operation.success", True)
                    return result

                except Exception as e:
                    span.set_attribute("operation.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

                finally:
                    execution_time = time.time() - start_time
                    span.set_attribute("operation.duration", execution_time)

        return wrapper

    return decorator


@contextmanager
def healthcare_span(operation_name: str, **attributes):
    """Context manager for creating healthcare AI spans"""

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(operation_name) as span:
        # Set default attributes
        span.set_attribute("service.name", "healthcare-ai")

        # Set custom attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)

        yield span


def init_observability(app=None):
    """Initialize all observability components"""

    # Configure logging
    configure_logging()

    # Setup tracing
    tracer = setup_tracing()

    # Instrument FastAPI if app is provided
    if app:
        FastAPIInstrumentor.instrument_app(app)

    # Create logger instance
    logger = HealthcareAILogger()

    # Log initialization
    logger.logger.info(
        "observability_initialized",
        service="healthcare-ai",
        tracing_enabled=True,
        metrics_enabled=True,
        structured_logging=True,
    )

    return logger, tracer


# Global instances (initialized when needed)
healthcare_logger = None
healthcare_tracer = None


def get_logger() -> HealthcareAILogger:
    """Get or create healthcare AI logger"""
    global healthcare_logger
    if healthcare_logger is None:
        configure_logging()
        healthcare_logger = HealthcareAILogger()
    return healthcare_logger


def get_tracer():
    """Get or create healthcare AI tracer"""
    global healthcare_tracer
    if healthcare_tracer is None:
        healthcare_tracer = setup_tracing()
    return healthcare_tracer
