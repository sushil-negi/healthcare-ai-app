#!/usr/bin/env python3
"""
Healthcare AI Metrics Exporter for Prometheus
Collects and exposes healthcare-specific metrics
"""

import logging
import os
import time
from typing import Any, Dict

import mlflow
import requests
import schedule
from prometheus_client import (Counter, Gauge, Histogram, Summary,
                               start_http_server)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
HEALTHCARE_AI_URL = os.getenv("HEALTHCARE_AI_URL", "http://localhost:8080")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
EXPORT_INTERVAL = int(os.getenv("EXPORT_INTERVAL", "10"))

# Healthcare-specific metrics
crisis_detection_rate = Gauge(
    "healthcare_ai_crisis_detection_rate",
    "Rate of successful crisis detections",
    ["model_version"],
)

model_accuracy = Gauge(
    "healthcare_ai_model_accuracy",
    "Current model accuracy",
    ["model_version", "model_stage"],
)

empathy_score = Gauge(
    "healthcare_ai_empathy_score",
    "Average empathy score in responses",
    ["model_version"],
)

response_time = Histogram(
    "healthcare_ai_response_time_seconds",
    "Response time in seconds",
    ["endpoint", "model_version"],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0],
)

request_count = Counter(
    "healthcare_ai_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

active_users = Gauge(
    "healthcare_ai_active_users", "Number of active users in the last 5 minutes"
)

category_distribution = Gauge(
    "healthcare_ai_category_requests",
    "Distribution of requests by healthcare category",
    ["category"],
)

# A/B testing metrics
ab_test_model_accuracy = Gauge(
    "ab_test_model_accuracy",
    "Model accuracy in A/B test",
    ["experiment_id", "model_id", "model_type"],
)

ab_test_model_crisis_detection = Gauge(
    "ab_test_model_crisis_detection_rate",
    "Crisis detection rate in A/B test",
    ["experiment_id", "model_id"],
)

# Data drift metrics
data_drift_score = Gauge(
    "healthcare_ai_data_drift_score",
    "Data drift score (0-1, higher means more drift)",
    ["feature", "drift_type"],
)


class HealthcareMetricsCollector:
    """Collects metrics from Healthcare AI service and MLflow"""

    def __init__(self):
        self.mlflow_client = None
        self.setup_mlflow()

    def setup_mlflow(self):
        """Initialize MLflow client"""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self.mlflow_client = mlflow.tracking.MlflowClient()
            logger.info(f"Connected to MLflow at {MLFLOW_TRACKING_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {e}")

    def collect_service_metrics(self):
        """Collect metrics from Healthcare AI service"""
        try:
            # Get health status
            health_response = requests.get(f"{HEALTHCARE_AI_URL}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()

                # Get metrics endpoint if available
                metrics_response = requests.get(
                    f"{HEALTHCARE_AI_URL}/metrics", timeout=5
                )
                if metrics_response.status_code == 200:
                    metrics_data = metrics_response.json()

                    # Update Prometheus metrics
                    if "crisis_detection_rate" in metrics_data:
                        crisis_detection_rate.labels(
                            model_version=metrics_data.get("model_version", "unknown")
                        ).set(metrics_data["crisis_detection_rate"])

                    if "accuracy" in metrics_data:
                        model_accuracy.labels(
                            model_version=metrics_data.get("model_version", "unknown"),
                            model_stage="production",
                        ).set(metrics_data["accuracy"])

                    if "empathy_score" in metrics_data:
                        empathy_score.labels(
                            model_version=metrics_data.get("model_version", "unknown")
                        ).set(metrics_data["empathy_score"])

                    if "active_users" in metrics_data:
                        active_users.set(metrics_data["active_users"])

                    # Category distribution
                    if "category_counts" in metrics_data:
                        for category, count in metrics_data["category_counts"].items():
                            category_distribution.labels(category=category).set(count)

                    logger.info("Successfully collected service metrics")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to collect service metrics: {e}")
        except Exception as e:
            logger.error(f"Unexpected error collecting metrics: {e}")

    def collect_mlflow_metrics(self):
        """Collect model metrics from MLflow"""
        if not self.mlflow_client:
            return

        try:
            # Get registered models
            models = self.mlflow_client.list_registered_models()

            for model in models:
                if "healthcare" in model.name.lower():
                    # Get latest version
                    latest_versions = self.mlflow_client.get_latest_versions(
                        model.name, stages=["Production", "Staging"]
                    )

                    for version in latest_versions:
                        # Get run data
                        run = self.mlflow_client.get_run(version.run_id)

                        # Extract metrics
                        if "accuracy" in run.data.metrics:
                            model_accuracy.labels(
                                model_version=version.version,
                                model_stage=version.current_stage,
                            ).set(run.data.metrics["accuracy"])

                        # Extract custom healthcare metrics from tags
                        if "crisis_detection_rate" in run.data.tags:
                            crisis_detection_rate.labels(
                                model_version=version.version
                            ).set(float(run.data.tags["crisis_detection_rate"]))

                        if "empathy_score" in run.data.tags:
                            empathy_score.labels(model_version=version.version).set(
                                float(run.data.tags["empathy_score"])
                            )

            logger.info("Successfully collected MLflow metrics")

        except Exception as e:
            logger.error(f"Failed to collect MLflow metrics: {e}")

    def collect_ab_test_metrics(self):
        """Collect A/B testing metrics if experiments are running"""
        try:
            # This would integrate with the A/B testing service
            # For now, we'll create placeholder logic
            ab_test_response = requests.get(
                f"{HEALTHCARE_AI_URL}/api/ab-tests/active", timeout=5
            )
            if ab_test_response.status_code == 200:
                experiments = ab_test_response.json()

                for exp in experiments:
                    ab_test_model_accuracy.labels(
                        experiment_id=exp["id"],
                        model_id=exp["model_a_id"],
                        model_type="control",
                    ).set(exp["model_a_metrics"].get("accuracy", 0))

                    ab_test_model_accuracy.labels(
                        experiment_id=exp["id"],
                        model_id=exp["model_b_id"],
                        model_type="test",
                    ).set(exp["model_b_metrics"].get("accuracy", 0))

                    # Crisis detection rates
                    ab_test_model_crisis_detection.labels(
                        experiment_id=exp["id"], model_id=exp["model_a_id"]
                    ).set(exp["model_a_metrics"].get("crisis_detection_rate", 0))

                    ab_test_model_crisis_detection.labels(
                        experiment_id=exp["id"], model_id=exp["model_b_id"]
                    ).set(exp["model_b_metrics"].get("crisis_detection_rate", 0))

        except requests.exceptions.RequestException:
            # A/B testing endpoint might not exist yet
            pass
        except Exception as e:
            logger.error(f"Failed to collect A/B test metrics: {e}")

    def collect_data_drift_metrics(self):
        """Collect data drift metrics"""
        try:
            drift_response = requests.get(
                f"{HEALTHCARE_AI_URL}/api/data-drift", timeout=5
            )
            if drift_response.status_code == 200:
                drift_data = drift_response.json()

                for feature, scores in drift_data.items():
                    if isinstance(scores, dict):
                        for drift_type, score in scores.items():
                            data_drift_score.labels(
                                feature=feature, drift_type=drift_type
                            ).set(score)

        except requests.exceptions.RequestException:
            # Data drift endpoint might not exist yet
            pass
        except Exception as e:
            logger.error(f"Failed to collect data drift metrics: {e}")

    def collect_all_metrics(self):
        """Collect all metrics"""
        logger.info("Collecting metrics...")
        self.collect_service_metrics()
        self.collect_mlflow_metrics()
        self.collect_ab_test_metrics()
        self.collect_data_drift_metrics()


def main():
    """Main function to run the metrics exporter"""
    # Start Prometheus metrics server
    start_http_server(9091)
    logger.info("Metrics exporter started on port 9091")

    # Initialize collector
    collector = HealthcareMetricsCollector()

    # Schedule metric collection
    schedule.every(EXPORT_INTERVAL).seconds.do(collector.collect_all_metrics)

    # Initial collection
    collector.collect_all_metrics()

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
