FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=kubernetes
ENV MONITORING_ENABLED=true
ENV MODEL_PATH=/app/models/healthcare_model.joblib

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "src/k8s_service.py"]