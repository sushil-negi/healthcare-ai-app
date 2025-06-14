"""
Kubernetes-optimized Healthcare AI Service
FastAPI-based service with monitoring, health checks, and production features
"""

import json
import logging
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import redis
except ImportError:
    redis = None
    print("Warning: Redis not available, running without caching")

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Prometheus metrics - create new registry to avoid conflicts
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: Prometheus client not available")

from pydantic import BaseModel

# Import healthcare AI engine
try:
    from src.healthcare_ai_engine import HealthcareAIEngine
except ImportError:
    from healthcare_ai_engine import HealthcareAIEngine

try:
    try:
        from src.observability import (
            get_logger,
            get_tracer,
            healthcare_span,
            init_observability,
            trace_healthcare_function,
        )
    except ImportError:
        from observability import (
            get_logger,
            get_tracer,
            healthcare_span,
            init_observability,
            trace_healthcare_function,
        )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    print("Warning: Observability features not available")

if PROMETHEUS_AVAILABLE:
    custom_registry = CollectorRegistry()

    request_count = Counter(
        "healthcare_ai_requests_total",
        "Total requests",
        ["method", "endpoint"],
        registry=custom_registry,
    )
    request_duration = Histogram(
        "healthcare_ai_request_duration_seconds",
        "Request duration",
        registry=custom_registry,
    )
    model_predictions = Counter(
        "healthcare_ai_predictions_total",
        "Total predictions",
        ["category", "method"],
        registry=custom_registry,
    )
    active_connections = Gauge(
        "healthcare_ai_active_connections",
        "Active connections",
        registry=custom_registry,
    )
    model_accuracy = Gauge(
        "healthcare_ai_model_accuracy",
        "Current model accuracy",
        registry=custom_registry,
    )
else:
    # Create dummy objects if Prometheus is not available
    request_count = None
    request_duration = None
    model_predictions = None
    active_connections = None
    model_accuracy = None

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Healthcare AI Service",
    description="Production Healthcare AI with MLOps integration",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    category: str
    confidence: float
    method: str
    generation_time: float
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    redis_connected: bool
    uptime_seconds: float
    version: str


# Global variables
healthcare_engine = None
redis_client = None
start_time = time.time()
observability_logger = None
tracer = None


def initialize_services():
    """Initialize healthcare AI engine and Redis connection"""
    global healthcare_engine, redis_client, observability_logger, tracer

    try:
        # Initialize observability first (if available)
        if OBSERVABILITY_AVAILABLE:
            observability_logger, tracer = init_observability(app)
        else:
            observability_logger, tracer = None, None

        # Initialize healthcare AI engine
        logger.info("Initializing Healthcare AI Engine...")
        healthcare_engine = HealthcareAIEngine()

        # Initialize Redis (optional, for caching)
        if redis:
            try:
                redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    decode_responses=True,
                    socket_timeout=5,
                )
                redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                redis_client = None
        else:
            logger.warning("Redis module not available")
            redis_client = None

        logger.info("Healthcare AI Service initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    initialize_services()


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add request timing and monitoring"""
    start_time = time.time()

    # Update active connections
    if active_connections:
        active_connections.inc()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Update metrics (if available)
        if request_count:
            request_count.labels(method=request.method, endpoint=request.url.path).inc()
        if request_duration:
            request_duration.observe(process_time)

        response.headers["X-Process-Time"] = str(process_time)
        return response

    finally:
        if active_connections:
            active_connections.dec()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kubernetes health check endpoint"""
    uptime = time.time() - start_time

    redis_status = False
    if redis_client:
        try:
            redis_client.ping()
            redis_status = True
        except:
            redis_status = False

    return HealthResponse(
        status="healthy",
        service="healthcare-ai",
        model_loaded=healthcare_engine is not None,
        redis_connected=redis_status,
        uptime_seconds=uptime,
        version="2.0.0",
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness check"""
    if healthcare_engine is None:
        raise HTTPException(status_code=503, detail="Healthcare AI engine not ready")

    return {"status": "ready", "service": "healthcare-ai"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(custom_registry), media_type="text/plain")


@app.post("/chat", response_model=ChatResponse)
@trace_healthcare_function("chat_request") if OBSERVABILITY_AVAILABLE else lambda x: x
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with monitoring"""
    if healthcare_engine is None:
        raise HTTPException(
            status_code=503, detail="Healthcare AI engine not initialized"
        )

    start_time = time.time()

    # Use observability if available, otherwise use a dummy context manager
    if OBSERVABILITY_AVAILABLE:
        span_context = healthcare_span(
            "chat_processing",
            message_length=len(request.message),
            user_id=request.user_id,
            session_id=request.session_id,
        )
    else:
        span_context = nullcontext()

    with span_context as span:

        try:
            # Check cache first (if Redis available)
            cache_key = None
            cached = False
            if redis_client:
                cache_key = f"chat:{hash(request.message.lower())}"
                try:
                    cached_response = redis_client.get(cache_key)
                    if cached_response:
                        cached = True
                        if span:
                            span.set_attribute("cache_hit", True)
                        logger.info("Returning cached response")
                        return ChatResponse.parse_raw(cached_response)
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")

            if span:
                span.set_attribute("cache_hit", False)

            # Generate response
            if OBSERVABILITY_AVAILABLE:
                inference_context = healthcare_span("model_inference")
            else:
                inference_context = nullcontext()

            with inference_context as inference_span:
                response_data = healthcare_engine.generate_response(request.message)

            generation_time = time.time() - start_time

            # Create response object
            chat_response = ChatResponse(
                response=response_data["response"],
                category=response_data.get("category", "general"),
                confidence=response_data.get("confidence", 0.0),
                method=response_data.get("method", "unknown"),
                generation_time=generation_time,
                model_version="2.0.0",
                timestamp=datetime.now().isoformat(),
            )

            # Add span attributes (if available)
            if span:
                span.set_attribute("response_category", chat_response.category)
                span.set_attribute("response_confidence", chat_response.confidence)
                span.set_attribute("response_time", generation_time)

            # Update metrics (if available)
            if model_predictions:
                model_predictions.labels(
                    category=chat_response.category, method=chat_response.method
                ).inc()

            # Cache response (if Redis available)
            if redis_client and cache_key:
                try:
                    redis_client.setex(
                        cache_key, 3600, chat_response.json()  # 1 hour cache
                    )
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")

            # Log request with observability logger
            if observability_logger:
                observability_logger.log_request(
                    category=chat_response.category,
                    confidence=chat_response.confidence,
                    response_time=generation_time,
                    method=chat_response.method,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    cached=cached,
                )

                # Check for performance issues
                observability_logger.log_performance_issue(
                    response_time=generation_time, category=chat_response.category
                )

            # Log request for monitoring
            logger.info(
                f"Chat request processed: {chat_response.category} in {generation_time:.3f}s"
            )

            return chat_response

        except Exception as e:
            if span:
                span.set_attribute("error", True)
                span.set_attribute("error_type", type(e).__name__)

            if observability_logger:
                observability_logger.log_error(
                    error=e,
                    context={
                        "user_id": request.user_id,
                        "session_id": request.session_id,
                        "message_length": len(request.message),
                    },
                )

            logger.error(f"Chat processing error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    if healthcare_engine is None:
        raise HTTPException(status_code=503, detail="Healthcare AI engine not ready")

    uptime = time.time() - start_time

    # Get engine stats if available
    engine_stats = {}
    if hasattr(healthcare_engine, "get_statistics"):
        engine_stats = healthcare_engine.get_statistics()

    # Get cache size from healthcare engine
    cache_size = 0
    if hasattr(healthcare_engine, "response_cache"):
        cache_size = len(healthcare_engine.response_cache)

    return {
        "service": "healthcare-ai",
        "version": "2.0.0",
        "uptime_seconds": uptime,
        "model_loaded": True,
        "redis_connected": redis_client is not None,
        "engine_stats": engine_stats,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "categories": 11,  # Add this field for e2e tests
        "category_list": [
            "adl",
            "senior_care",
            "mental_health",
            "respite_care",
            "disabilities",
            "general",
            "crisis",
            "medication",
            "mobility",
            "nutrition",
            "social",
        ],
        "total_requests": engine_stats.get("total_requests", 0),
        "cache_size": cache_size,
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Healthcare AI",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    workers = int(os.getenv("WORKERS", 1))
    log_level = os.getenv("LOG_LEVEL", "info")

    logger.info(f"Starting Healthcare AI Service on {host}:{port}")

    # Run with uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
    )
