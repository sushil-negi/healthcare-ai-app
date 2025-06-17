#!/usr/bin/env python3
"""
Model Drift Detection System for Healthcare AI
Monitors for data drift and model performance degradation
"""

import json
import logging
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy import stats

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDriftDetector:
    def __init__(self):
        self.baseline_stats = self.load_baseline_statistics()
        self.drift_thresholds = {
            "ks_statistic": 0.15,  # Kolmogorov-Smirnov test threshold
            "psi": 0.2,  # Population Stability Index threshold
            "accuracy_drop": 0.05,  # 5% accuracy drop
            "response_time_increase": 0.5,  # 50% response time increase
            "crisis_detection_drop": 0.02,  # 2% drop in crisis detection
        }
        self.monitoring_window = timedelta(hours=24)

    def load_baseline_statistics(self) -> Dict:
        """Load baseline statistics from model training"""
        return {
            "query_length_distribution": {
                "mean": 45.3,
                "std": 18.7,
                "percentiles": [10, 25, 50, 75, 90],
            },
            "category_distribution": {
                "medication": 0.25,
                "mobility": 0.20,
                "mental_health": 0.15,
                "caregiver": 0.15,
                "general": 0.25,
            },
            "response_metrics": {
                "accuracy": 0.95,
                "response_time_ms": 120,
                "user_satisfaction": 0.89,
                "crisis_detection_rate": 0.99,
            },
            "vocabulary_stats": {"unique_tokens": 15000, "avg_tokens_per_query": 12},
        }

    def calculate_ks_statistic(
        self, baseline_data: List[float], current_data: List[float]
    ) -> Tuple[float, float]:
        """Calculate Kolmogorov-Smirnov statistic for distribution drift"""
        if not baseline_data or not current_data:
            return 0.0, 1.0

        statistic, p_value = stats.ks_2samp(baseline_data, current_data)
        return statistic, p_value

    def calculate_psi(
        self, baseline_probs: Dict[str, float], current_probs: Dict[str, float]
    ) -> float:
        """Calculate Population Stability Index for categorical drift"""
        psi = 0.0

        for category in baseline_probs:
            baseline_p = baseline_probs.get(category, 0.001)  # Avoid log(0)
            current_p = current_probs.get(category, 0.001)

            # PSI formula: (current% - baseline%) * ln(current% / baseline%)
            psi += (current_p - baseline_p) * np.log(current_p / baseline_p)

        return psi

    def detect_data_drift(self, recent_queries: List[Dict]) -> Dict:
        """Detect drift in input data distribution"""
        logger.info("🔍 Detecting data drift...")

        drift_results = {"drift_detected": False, "drift_scores": {}, "drift_types": []}

        if not recent_queries:
            logger.warning("No recent queries to analyze")
            return drift_results

        # 1. Query Length Distribution Drift
        query_lengths = [len(q.get("query", "").split()) for q in recent_queries]
        if query_lengths:
            baseline_lengths = np.random.normal(
                self.baseline_stats["query_length_distribution"]["mean"],
                self.baseline_stats["query_length_distribution"]["std"],
                1000,
            )

            ks_stat, p_value = self.calculate_ks_statistic(
                baseline_lengths, query_lengths
            )
            drift_results["drift_scores"]["query_length_ks"] = ks_stat

            if ks_stat > self.drift_thresholds["ks_statistic"]:
                drift_results["drift_detected"] = True
                drift_results["drift_types"].append(
                    f"Query length drift (KS={ks_stat:.3f})"
                )

        # 2. Category Distribution Drift
        category_counts = defaultdict(int)
        for q in recent_queries:
            category = q.get("category", "general")
            category_counts[category] += 1

        total_queries = len(recent_queries)
        current_category_dist = {
            cat: count / total_queries for cat, count in category_counts.items()
        }

        psi = self.calculate_psi(
            self.baseline_stats["category_distribution"], current_category_dist
        )
        drift_results["drift_scores"]["category_psi"] = psi

        if psi > self.drift_thresholds["psi"]:
            drift_results["drift_detected"] = True
            drift_results["drift_types"].append(
                f"Category distribution drift (PSI={psi:.3f})"
            )

        # 3. New Vocabulary Detection
        current_tokens = set()
        for q in recent_queries:
            tokens = q.get("query", "").lower().split()
            current_tokens.update(tokens)

        new_token_ratio = (
            len(current_tokens)
            / self.baseline_stats["vocabulary_stats"]["unique_tokens"]
        )
        if new_token_ratio > 1.2:  # 20% more unique tokens
            drift_results["drift_detected"] = True
            drift_results["drift_types"].append(
                f"Vocabulary expansion ({new_token_ratio:.1%})"
            )

        logger.info(f"✅ Data drift analysis complete: {drift_results}")
        return drift_results

    def detect_performance_drift(self, recent_metrics: List[Dict]) -> Dict:
        """Detect drift in model performance metrics"""
        logger.info("📊 Detecting performance drift...")

        drift_results = {
            "drift_detected": False,
            "performance_scores": {},
            "degradation_types": [],
        }

        if not recent_metrics:
            logger.warning("No recent metrics to analyze")
            return drift_results

        # Calculate average recent metrics
        avg_metrics = {
            "accuracy": np.mean([m.get("accuracy", 0) for m in recent_metrics]),
            "response_time_ms": np.mean(
                [m.get("response_time_ms", 0) for m in recent_metrics]
            ),
            "user_satisfaction": np.mean(
                [m.get("user_satisfaction", 0) for m in recent_metrics]
            ),
            "crisis_detection_rate": np.mean(
                [m.get("crisis_detection_rate", 0) for m in recent_metrics]
            ),
        }

        baseline = self.baseline_stats["response_metrics"]

        # 1. Accuracy Degradation
        accuracy_drop = baseline["accuracy"] - avg_metrics["accuracy"]
        drift_results["performance_scores"]["accuracy_drop"] = accuracy_drop

        if accuracy_drop > self.drift_thresholds["accuracy_drop"]:
            drift_results["drift_detected"] = True
            drift_results["degradation_types"].append(
                f"Accuracy degradation ({avg_metrics['accuracy']:.1%} vs {baseline['accuracy']:.1%})"
            )

        # 2. Response Time Degradation
        rt_increase = (
            avg_metrics["response_time_ms"] - baseline["response_time_ms"]
        ) / baseline["response_time_ms"]
        drift_results["performance_scores"]["response_time_increase"] = rt_increase

        if rt_increase > self.drift_thresholds["response_time_increase"]:
            drift_results["drift_detected"] = True
            drift_results["degradation_types"].append(
                f"Response time increase ({avg_metrics['response_time_ms']:.0f}ms vs {baseline['response_time_ms']:.0f}ms)"
            )

        # 3. Crisis Detection Degradation (Critical!)
        crisis_drop = (
            baseline["crisis_detection_rate"] - avg_metrics["crisis_detection_rate"]
        )
        drift_results["performance_scores"]["crisis_detection_drop"] = crisis_drop

        if crisis_drop > self.drift_thresholds["crisis_detection_drop"]:
            drift_results["drift_detected"] = True
            drift_results["degradation_types"].append(
                f"⚠️ CRITICAL: Crisis detection degradation ({avg_metrics['crisis_detection_rate']:.1%} vs {baseline['crisis_detection_rate']:.1%})"
            )

        logger.info(f"✅ Performance drift analysis complete: {drift_results}")
        return drift_results

    def generate_drift_report(self, data_drift: Dict, performance_drift: Dict) -> Dict:
        """Generate comprehensive drift detection report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_drift_detected": data_drift["drift_detected"]
            or performance_drift["drift_detected"],
            "data_drift": data_drift,
            "performance_drift": performance_drift,
            "recommendations": [],
        }

        # Generate recommendations
        if data_drift["drift_detected"]:
            report["recommendations"].append("Consider retraining with recent data")

        if performance_drift["drift_detected"]:
            if "crisis_detection_drop" in str(performance_drift["degradation_types"]):
                report["recommendations"].append(
                    "🚨 URGENT: Review crisis detection system immediately"
                )
            report["recommendations"].append("Trigger A/B test with retrained model")

        if report["overall_drift_detected"]:
            report["recommendations"].append("Review recent changes in user behavior")
            report["recommendations"].append("Check for seasonal patterns")

        return report

    def monitor_real_time(self, stream_data: Dict):
        """Real-time drift monitoring (called by streaming system)"""
        # This would be called by a streaming system (Kafka, etc.)
        # For demo purposes, we'll simulate batch processing

        logger.info("🔄 Real-time drift monitoring active...")

        # Check if we have enough data for analysis
        if hasattr(self, "recent_buffer"):
            self.recent_buffer.append(stream_data)
        else:
            self.recent_buffer = [stream_data]

        # Analyze every 100 queries
        if len(self.recent_buffer) >= 100:
            data_drift = self.detect_data_drift(self.recent_buffer)

            # Extract metrics from buffer
            metrics = [
                {"accuracy": d.get("accuracy", 0.95)} for d in self.recent_buffer
            ]
            performance_drift = self.detect_performance_drift(metrics)

            if data_drift["drift_detected"] or performance_drift["drift_detected"]:
                logger.warning("⚠️ DRIFT DETECTED - Sending alert!")
                self.send_drift_alert(data_drift, performance_drift)

            # Reset buffer
            self.recent_buffer = []

    def send_drift_alert(self, data_drift: Dict, performance_drift: Dict):
        """Send drift detection alerts"""
        alert = {
            "severity": "HIGH" if performance_drift["drift_detected"] else "MEDIUM",
            "title": "Model Drift Detected",
            "data_drift": data_drift["drift_types"],
            "performance_drift": performance_drift["degradation_types"],
            "timestamp": datetime.now().isoformat(),
        }

        logger.error(f"🚨 DRIFT ALERT: {json.dumps(alert, indent=2)}")

        # In production, this would:
        # - Send to alerting system (PagerDuty, Slack)
        # - Trigger auto-retraining pipeline
        # - Create incident ticket
        # - Notify ML team


def demonstrate_drift_detection():
    """Demonstrate drift detection capabilities"""
    detector = ModelDriftDetector()

    print("\n" + "=" * 60)
    print("🔍 MODEL DRIFT DETECTION DEMO")
    print("=" * 60 + "\n")

    # Simulate normal queries
    print("1️⃣ Testing with normal data distribution...")
    normal_queries = [
        {"query": "help with medication schedule", "category": "medication"},
        {"query": "mobility aids for elderly", "category": "mobility"},
        {"query": "feeling anxious about caregiving", "category": "mental_health"},
    ] * 30

    normal_metrics = [
        {"accuracy": 0.94, "response_time_ms": 125, "crisis_detection_rate": 0.99}
    ] * 10

    data_drift = detector.detect_data_drift(normal_queries)
    perf_drift = detector.detect_performance_drift(normal_metrics)

    print(f"   Data drift detected: {data_drift['drift_detected']}")
    print(f"   Performance drift detected: {perf_drift['drift_detected']}")

    # Simulate drift scenario
    print("\n2️⃣ Testing with drifted data...")
    drifted_queries = [
        {
            "query": "covid vaccine side effects medication interaction",
            "category": "medication",
        },
        {"query": "long covid fatigue management tips", "category": "general"},
        {"query": "pandemic isolation mental health", "category": "mental_health"},
    ] * 30

    drifted_metrics = [
        {"accuracy": 0.88, "response_time_ms": 200, "crisis_detection_rate": 0.96}
    ] * 10

    data_drift = detector.detect_data_drift(drifted_queries)
    perf_drift = detector.detect_performance_drift(drifted_metrics)

    print(f"   Data drift detected: {data_drift['drift_detected']}")
    print(f"   Drift types: {data_drift['drift_types']}")
    print(f"   Performance drift detected: {perf_drift['drift_detected']}")
    print(f"   Degradation types: {perf_drift['degradation_types']}")

    # Generate report
    print("\n3️⃣ Generating drift report...")
    report = detector.generate_drift_report(data_drift, perf_drift)
    print(f"\n📋 Drift Detection Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    demonstrate_drift_detection()
