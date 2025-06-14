"""
Healthcare AI Model Monitoring System
Real-time monitoring for model performance, data drift, and safety metrics
"""

import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Prometheus metrics
model_predictions_total = Counter(
    "healthcare_ai_predictions_total", "Total predictions made", ["category", "method"]
)
prediction_latency = Histogram(
    "healthcare_ai_prediction_latency_seconds", "Prediction latency"
)
model_accuracy = Gauge("healthcare_ai_model_accuracy", "Current model accuracy")
crisis_detection_rate = Gauge(
    "healthcare_ai_crisis_detection_rate", "Crisis detection accuracy"
)
data_drift_score = Gauge("healthcare_ai_data_drift_score", "Data drift score (0-1)")
response_quality_score = Gauge(
    "healthcare_ai_response_quality_score", "Response quality score"
)
active_users = Gauge("healthcare_ai_active_users", "Number of active users")
error_rate = Gauge("healthcare_ai_error_rate", "Error rate percentage")


class HealthcareModelMonitor:
    """Real-time monitoring for healthcare AI model"""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        mlflow_uri: str = "http://localhost:5001",
        monitoring_interval: int = 60,
    ):

        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self.mlflow_uri = mlflow_uri
        self.monitoring_interval = monitoring_interval
        self.baseline_vectorizer = None
        self.baseline_queries = []
        self.running = True

        # Healthcare-specific thresholds
        self.thresholds = {
            "crisis_detection_min": 0.99,
            "accuracy_min": 0.85,
            "data_drift_max": 0.3,
            "response_quality_min": 0.80,
            "error_rate_max": 0.05,
        }

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load baseline data for drift detection
        self._load_baseline_data()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Received shutdown signal. Stopping monitor...")
        self.running = False

    def _load_baseline_data(self):
        """Load baseline training data for drift detection"""
        try:
            # Load training data (assuming it's available)
            with open("/data/combined_healthcare_training_data.json", "r") as f:
                baseline_data = json.load(f)

            self.baseline_queries = [item["query"] for item in baseline_data]

            # Create baseline TF-IDF for drift detection
            self.baseline_vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english"
            )
            self.baseline_vectorizer.fit(self.baseline_queries)

            self.logger.info(
                f"Loaded {len(self.baseline_queries)} baseline queries for drift detection"
            )

        except Exception as e:
            self.logger.error(f"Failed to load baseline data: {e}")
            self.baseline_queries = []

    def record_prediction(self, query: str, prediction: Dict, latency: float):
        """Record a prediction for monitoring"""
        try:
            # Update Prometheus metrics
            model_predictions_total.labels(
                category=prediction.get("category", "unknown"),
                method=prediction.get("method", "unknown"),
            ).inc()

            prediction_latency.observe(latency)

            # Store in Redis for batch analysis
            prediction_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "prediction": prediction,
                "latency": latency,
            }

            # Store recent predictions (sliding window)
            key = f"predictions:{datetime.now().strftime('%Y-%m-%d-%H')}"
            self.redis_client.lpush(key, json.dumps(prediction_data))
            self.redis_client.expire(key, 86400)  # Expire after 24 hours

            # Store for immediate analysis
            self.redis_client.lpush("recent_predictions", json.dumps(prediction_data))
            self.redis_client.ltrim("recent_predictions", 0, 999)  # Keep last 1000

        except Exception as e:
            self.logger.error(f"Failed to record prediction: {e}")

    def record_error(self, query: str, error: str):
        """Record an error for monitoring"""
        try:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "error": error,
            }

            self.redis_client.lpush("errors", json.dumps(error_data))
            self.redis_client.ltrim("errors", 0, 499)  # Keep last 500 errors

        except Exception as e:
            self.logger.error(f"Failed to record error: {e}")

    def calculate_data_drift(self, recent_queries: List[str]) -> float:
        """Calculate data drift using cosine similarity"""
        if not self.baseline_vectorizer or not recent_queries:
            return 0.0

        try:
            # Vectorize recent queries
            recent_vectors = self.baseline_vectorizer.transform(recent_queries)
            baseline_vectors = self.baseline_vectorizer.transform(
                self.baseline_queries[: len(recent_queries)]
            )

            # Calculate average cosine similarity
            similarities = []
            for i in range(min(len(recent_queries), len(self.baseline_queries))):
                sim = cosine_similarity(
                    recent_vectors[i : i + 1], baseline_vectors[i : i + 1]
                )[0][0]
                similarities.append(sim)

            # Drift score = 1 - average similarity
            drift_score = 1.0 - np.mean(similarities) if similarities else 0.0
            return max(0.0, min(1.0, drift_score))

        except Exception as e:
            self.logger.error(f"Failed to calculate data drift: {e}")
            return 0.0

    def calculate_crisis_detection_rate(self, predictions: List[Dict]) -> float:
        """Calculate crisis detection effectiveness"""
        try:
            crisis_queries = []
            crisis_predictions = []

            for pred in predictions:
                query = pred["query"].lower()
                # Check if query contains crisis indicators
                crisis_indicators = [
                    "suicide",
                    "kill myself",
                    "end my life",
                    "want to die",
                    "hurt myself",
                ]
                if any(indicator in query for indicator in crisis_indicators):
                    crisis_queries.append(query)
                    # Check if model detected crisis
                    detected_crisis = (
                        pred["prediction"].get("category") == "crisis_mental_health"
                        or "crisis" in pred["prediction"].get("response", "").lower()
                        or "988" in pred["prediction"].get("response", "")
                    )
                    crisis_predictions.append(detected_crisis)

            if crisis_queries:
                return sum(crisis_predictions) / len(crisis_predictions)
            return 1.0  # No crisis queries = perfect detection

        except Exception as e:
            self.logger.error(f"Failed to calculate crisis detection rate: {e}")
            return 0.0

    def calculate_response_quality(self, predictions: List[Dict]) -> float:
        """Calculate response quality score"""
        try:
            quality_scores = []

            for pred in predictions:
                response = pred["prediction"].get("response", "")
                score = 0.0

                # Check for numbered steps
                if any(f"{i})" in response for i in range(1, 6)):
                    score += 0.3

                # Check for disclaimer
                if "⚠️" in response or "consult" in response.lower():
                    score += 0.3

                # Check for specificity (length and detail)
                if len(response) > 100:
                    score += 0.2

                # Check for actionable advice
                action_words = [
                    "install",
                    "use",
                    "consider",
                    "try",
                    "practice",
                    "contact",
                ]
                if any(word in response.lower() for word in action_words):
                    score += 0.2

                quality_scores.append(min(1.0, score))

            return np.mean(quality_scores) if quality_scores else 0.0

        except Exception as e:
            self.logger.error(f"Failed to calculate response quality: {e}")
            return 0.0

    def analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent model performance"""
        try:
            # Get recent predictions
            recent_data = self.redis_client.lrange("recent_predictions", 0, 999)
            predictions = [json.loads(data) for data in recent_data]

            if not predictions:
                return {"error": "No recent predictions found"}

            # Calculate metrics
            queries = [p["query"] for p in predictions]
            latencies = [p["latency"] for p in predictions]

            # Data drift
            drift_score = self.calculate_data_drift(queries)
            data_drift_score.set(drift_score)

            # Crisis detection rate
            crisis_rate = self.calculate_crisis_detection_rate(predictions)
            crisis_detection_rate.set(crisis_rate)

            # Response quality
            quality_score = self.calculate_response_quality(predictions)
            response_quality_score.set(quality_score)

            # Error rate
            recent_errors = self.redis_client.lrange("errors", 0, 99)
            error_rate_value = len(recent_errors) / max(len(predictions), 1)
            error_rate.set(error_rate_value)

            # Performance metrics
            avg_latency = np.mean(latencies) if latencies else 0.0

            metrics = {
                "data_drift_score": drift_score,
                "crisis_detection_rate": crisis_rate,
                "response_quality_score": quality_score,
                "error_rate": error_rate_value,
                "avg_latency": avg_latency,
                "total_predictions": len(predictions),
            }

            # Check for alerts
            alerts = self._check_alerts(metrics)
            if alerts:
                self._send_alerts(alerts)

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to analyze performance: {e}")
            return {"error": str(e)}

    def _check_alerts(self, metrics: Dict[str, float]) -> List[str]:
        """Check if any metrics exceed thresholds"""
        alerts = []

        if metrics["crisis_detection_rate"] < self.thresholds["crisis_detection_min"]:
            alerts.append(
                f"CRITICAL: Crisis detection rate below threshold: {metrics['crisis_detection_rate']:.3f}"
            )

        if metrics["data_drift_score"] > self.thresholds["data_drift_max"]:
            alerts.append(
                f"WARNING: High data drift detected: {metrics['data_drift_score']:.3f}"
            )

        if metrics["response_quality_score"] < self.thresholds["response_quality_min"]:
            alerts.append(
                f"WARNING: Response quality below threshold: {metrics['response_quality_score']:.3f}"
            )

        if metrics["error_rate"] > self.thresholds["error_rate_max"]:
            alerts.append(f"WARNING: High error rate: {metrics['error_rate']:.3f}")

        return alerts

    def _send_alerts(self, alerts: List[str]):
        """Send alerts to monitoring systems"""
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert}")
            # Here you would integrate with Slack, PagerDuty, etc.

    def start_monitoring(self):
        """Start the monitoring loop"""
        self.logger.info(
            f"Starting healthcare AI model monitoring (interval: {self.monitoring_interval}s)"
        )

        # Start Prometheus metrics server
        start_http_server(8000)
        self.logger.info("Prometheus metrics server started on port 8000")

        while self.running:
            try:
                metrics = self.analyze_recent_performance()
                self.logger.info(f"Monitoring metrics: {metrics}")
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

        self.logger.info("Healthcare AI model monitoring stopped")


def main():
    """Main function to start monitoring"""
    monitor = HealthcareModelMonitor()
    monitor.start_monitoring()


if __name__ == "__main__":
    main()
