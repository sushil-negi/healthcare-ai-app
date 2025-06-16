# Optimized single-stage build for faster CI
FROM python:3.11-slim

LABEL maintainer="Cirruslabs MLOps Team"
LABEL service="healthcare-ai"
LABEL stage="production"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements and install Python packages (optimized versions)
COPY requirements.advanced.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.advanced.txt

# Copy application code (optimized layer)
COPY src/ ./src/
COPY service.py .

# Create data directory (training data may be mounted via volume)
RUN mkdir -p /app/data

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/artifacts && \
    chown appuser:appuser /app/artifacts

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the healthcare AI service
CMD ["python", "service.py"]
