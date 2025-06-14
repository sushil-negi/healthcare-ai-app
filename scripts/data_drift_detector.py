#!/usr/bin/env python3
"""
Data Drift Detection for Healthcare AI
Monitors input data distribution changes over time
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareDataDriftDetector:
    """Detect data drift in healthcare AI inputs"""

    def __init__(
        self,
        reference_data_path: str,
        drift_threshold: float = 0.1,
        significance_level: float = 0.05,
    ):
        """
        Initialize drift detector

        Args:
            reference_data_path: Path to reference/training data
            drift_threshold: Threshold for drift detection (0-1)
            significance_level: Statistical significance level
        """
        self.reference_data_path = reference_data_path
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level

        # Load reference data
        self.reference_data = self._load_reference_data()
        self.reference_stats = self._compute_reference_stats()

        # Initialize TF-IDF for text similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=min(
                1000, len(self.reference_data) * 10
            ),  # Adjust for small datasets
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,  # Allow rare words for small datasets
            max_df=0.95,
        )

        # Fit on reference data
        reference_texts = [item.get("text", "") for item in self.reference_data]
        # Filter out empty texts
        reference_texts = [text for text in reference_texts if text.strip()]

        if len(reference_texts) == 0:
            logger.warning("No valid text data found in reference dataset")
            self.reference_tfidf = None
        else:
            try:
                self.reference_tfidf = self.tfidf_vectorizer.fit_transform(
                    reference_texts
                )
            except ValueError as e:
                logger.warning(f"TF-IDF fitting failed: {e}. Using simpler approach.")
                # Fallback: use simple word-based approach
                self.tfidf_vectorizer = None
                self.reference_tfidf = None

        logger.info(
            f"Initialized drift detector with {len(self.reference_data)} reference samples"
        )

    def _load_reference_data(self) -> List[Dict]:
        """Load reference training data"""
        try:
            with open(self.reference_data_path, "r") as f:
                data = json.load(f)

            # Handle different data formats
            if isinstance(data, list):
                # Filter out non-dict items (like "texts", "metadata")
                valid_items = []
                for item in data:
                    if isinstance(item, dict):
                        # Handle different text field names
                        text_field = None
                        if "user_query" in item:
                            text_field = item["user_query"]
                        elif "query" in item:
                            text_field = item["query"]
                        elif "text" in item:
                            text_field = item["text"]

                        if text_field and "category" in item:
                            valid_items.append(
                                {"text": text_field, "category": item["category"]}
                            )
                return valid_items
            else:
                return [data] if isinstance(data, dict) else []

        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            return []

    def _compute_reference_stats(self) -> Dict:
        """Compute reference statistics"""
        if not self.reference_data:
            return {}

        # Text length statistics
        text_lengths = [len(item.get("text", "")) for item in self.reference_data]

        # Category distribution
        categories = [item.get("category", "unknown") for item in self.reference_data]
        category_counts = pd.Series(categories).value_counts(normalize=True)

        # Word count statistics
        word_counts = [
            len(item.get("text", "").split()) for item in self.reference_data
        ]

        stats = {
            "text_length": {
                "mean": np.mean(text_lengths),
                "std": np.std(text_lengths),
                "min": np.min(text_lengths),
                "max": np.max(text_lengths),
                "median": np.median(text_lengths),
            },
            "word_count": {
                "mean": np.mean(word_counts),
                "std": np.std(word_counts),
                "min": np.min(word_counts),
                "max": np.max(word_counts),
                "median": np.median(word_counts),
            },
            "category_distribution": category_counts.to_dict(),
            "total_samples": len(self.reference_data),
        }

        return stats

    def detect_text_drift(self, current_data: List[Dict]) -> Dict:
        """Detect drift in text characteristics"""
        if not current_data:
            return {"drift_detected": False, "reason": "No current data"}

        # Extract texts
        current_texts = [item.get("text", "") for item in current_data]

        # Compute current statistics
        current_text_lengths = [len(text) for text in current_texts]
        current_word_counts = [len(text.split()) for text in current_texts]

        # Statistical tests for text length
        length_statistic, length_pvalue = stats.ks_2samp(
            [len(item.get("text", "")) for item in self.reference_data],
            current_text_lengths,
        )

        # Statistical tests for word count
        word_statistic, word_pvalue = stats.ks_2samp(
            [len(item.get("text", "").split()) for item in self.reference_data],
            current_word_counts,
        )

        # Semantic similarity using TF-IDF
        if self.tfidf_vectorizer is not None and self.reference_tfidf is not None:
            try:
                current_tfidf = self.tfidf_vectorizer.transform(current_texts)
                similarity_scores = cosine_similarity(
                    current_tfidf, self.reference_tfidf
                )
                avg_similarity = np.mean(similarity_scores.max(axis=1))
            except Exception as e:
                logger.warning(f"TF-IDF similarity calculation failed: {e}")
                avg_similarity = 0.8  # Default neutral similarity
        else:
            # Fallback: simple word overlap
            ref_words = set()
            for item in self.reference_data:
                ref_words.update(item.get("text", "").lower().split())

            if ref_words:
                current_words = set()
                for text in current_texts:
                    current_words.update(text.lower().split())

                overlap = len(ref_words & current_words)
                avg_similarity = (
                    overlap / len(ref_words | current_words)
                    if (ref_words | current_words)
                    else 0.5
                )
            else:
                avg_similarity = 0.5

        # Check for drift
        length_drift = bool(length_pvalue < self.significance_level)
        word_drift = bool(word_pvalue < self.significance_level)
        semantic_drift = bool(avg_similarity < (1 - self.drift_threshold))

        drift_detected = bool(length_drift or word_drift or semantic_drift)

        return {
            "drift_detected": drift_detected,
            "length_drift": length_drift,
            "word_drift": word_drift,
            "semantic_drift": semantic_drift,
            "length_pvalue": length_pvalue,
            "word_pvalue": word_pvalue,
            "avg_similarity": avg_similarity,
            "current_stats": {
                "text_length_mean": np.mean(current_text_lengths),
                "word_count_mean": np.mean(current_word_counts),
                "sample_count": len(current_data),
            },
        }

    def detect_category_drift(self, current_data: List[Dict]) -> Dict:
        """Detect drift in category distribution"""
        if not current_data:
            return {"drift_detected": False, "reason": "No current data"}

        # Get current category distribution
        current_categories = [item.get("category", "unknown") for item in current_data]
        current_dist = pd.Series(current_categories).value_counts(normalize=True)

        # Get reference distribution
        ref_dist = pd.Series(self.reference_stats["category_distribution"])

        # Align distributions (handle missing categories)
        all_categories = set(current_dist.index) | set(ref_dist.index)

        ref_aligned = []
        current_aligned = []

        for cat in all_categories:
            ref_aligned.append(ref_dist.get(cat, 0))
            current_aligned.append(current_dist.get(cat, 0))

        # Chi-square test for category distribution
        try:
            # Convert to counts for chi-square test
            ref_counts = [
                int(p * self.reference_stats["total_samples"]) for p in ref_aligned
            ]
            current_counts = [int(p * len(current_data)) for p in current_aligned]

            chi2_stat, chi2_pvalue = stats.chisquare(current_counts, ref_counts)

            category_drift = chi2_pvalue < self.significance_level

        except Exception as e:
            logger.warning(f"Could not perform chi-square test: {e}")
            # Fallback to simple comparison
            max_diff = max(abs(c - r) for c, r in zip(current_aligned, ref_aligned))
            category_drift = bool(max_diff > self.drift_threshold)
            chi2_pvalue = None

        return {
            "drift_detected": category_drift,
            "chi2_pvalue": chi2_pvalue,
            "reference_distribution": dict(zip(all_categories, ref_aligned)),
            "current_distribution": dict(zip(all_categories, current_aligned)),
            "max_difference": max(
                abs(c - r) for c, r in zip(current_aligned, ref_aligned)
            ),
        }

    def detect_comprehensive_drift(self, current_data: List[Dict]) -> Dict:
        """Comprehensive drift detection"""

        text_drift = self.detect_text_drift(current_data)
        category_drift = self.detect_category_drift(current_data)

        # Overall drift assessment
        overall_drift = text_drift.get("drift_detected", False) or category_drift.get(
            "drift_detected", False
        )

        # Severity assessment
        severity = "low"
        if text_drift.get("semantic_drift", False):
            severity = "high"
        elif text_drift.get("length_drift", False) and category_drift.get(
            "drift_detected", False
        ):
            severity = "medium"

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_drift_detected": overall_drift,
            "severity": severity,
            "text_drift": text_drift,
            "category_drift": category_drift,
            "recommendations": self._generate_recommendations(
                text_drift, category_drift
            ),
        }

    def _generate_recommendations(
        self, text_drift: Dict, category_drift: Dict
    ) -> List[str]:
        """Generate recommendations based on drift results"""
        recommendations = []

        if text_drift.get("semantic_drift", False):
            recommendations.append(
                "Significant semantic drift detected - consider retraining model"
            )
            recommendations.append(
                "Review recent user queries for new topics or terminology"
            )

        if text_drift.get("length_drift", False):
            recommendations.append(
                "Text length distribution has changed - review input preprocessing"
            )

        if category_drift.get("drift_detected", False):
            recommendations.append(
                "Category distribution has shifted - monitor classification accuracy"
            )
            recommendations.append(
                "Consider collecting more data for underrepresented categories"
            )

        if text_drift.get("avg_similarity", 1.0) < 0.7:
            recommendations.append(
                "Low semantic similarity - urgent model update recommended"
            )

        if not recommendations:
            recommendations.append(
                "No significant drift detected - continue monitoring"
            )

        return recommendations


def collect_recent_data(hours: int = 24, api_endpoint: str = None) -> List[Dict]:
    """Collect recent production data for drift analysis"""

    # For demo purposes, we'll simulate this by reading from logs
    # In production, this would query your logging/monitoring system

    try:
        # Example: reading from application logs
        log_file = "/logs/healthcare-ai/production.log"
        if os.path.exists(log_file):
            recent_data = []
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            with open(log_file, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        if "user_query" in log_entry and "category" in log_entry:
                            # Convert to expected format
                            recent_data.append(
                                {
                                    "text": log_entry["user_query"],
                                    "category": log_entry["category"],
                                    "timestamp": log_entry.get("timestamp"),
                                }
                            )
                    except:
                        continue

            return recent_data

        # Fallback: simulate some production data
        logger.warning("No production logs found, using simulated data")
        return [
            {"text": "I need help with mobility aids", "category": "adl"},
            {
                "text": "My elderly father is having trouble walking",
                "category": "senior_care",
            },
            {
                "text": "I feel very anxious about my health",
                "category": "mental_health",
            },
            {"text": "Need respite care options", "category": "respite_care"},
        ]

    except Exception as e:
        logger.error(f"Error collecting recent data: {e}")
        return []


def send_drift_alert(drift_results: Dict, webhook_url: str = None):
    """Send drift alert to monitoring system"""

    if not drift_results["overall_drift_detected"]:
        return

    alert_message = {
        "alert_type": "data_drift",
        "severity": drift_results["severity"],
        "timestamp": drift_results["timestamp"],
        "message": f"Data drift detected (severity: {drift_results['severity']})",
        "recommendations": drift_results["recommendations"],
    }

    # Log alert
    logger.warning(f"DATA DRIFT ALERT: {alert_message}")

    # Send to webhook if configured
    if webhook_url:
        try:
            response = requests.post(webhook_url, json=alert_message, timeout=10)
            response.raise_for_status()
            logger.info("Drift alert sent to webhook")
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")


def main():
    parser = argparse.ArgumentParser(description="Healthcare AI Data Drift Detection")
    parser.add_argument(
        "--reference-data",
        default="data/combined_healthcare_training_data.json",
        help="Path to reference training data",
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="Hours of recent data to analyze"
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.1,
        help="Drift detection threshold (0-1)",
    )
    parser.add_argument("--output", help="Output file for drift report")
    parser.add_argument("--webhook-url", help="Webhook URL for alerts")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize drift detector
    detector = HealthcareDataDriftDetector(
        reference_data_path=args.reference_data, drift_threshold=args.drift_threshold
    )

    # Collect recent data
    logger.info(f"Collecting recent data ({args.hours} hours)")
    recent_data = collect_recent_data(hours=args.hours)

    if not recent_data:
        logger.warning("No recent data found for drift analysis")
        return

    logger.info(f"Analyzing {len(recent_data)} recent samples for drift")

    # Perform drift detection
    drift_results = detector.detect_comprehensive_drift(recent_data)

    # Output results
    print(json.dumps(drift_results, indent=2))

    # Save to file if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(drift_results, f, indent=2)
        logger.info(f"Drift report saved to {args.output}")

    # Send alerts if drift detected
    if drift_results["overall_drift_detected"]:
        send_drift_alert(drift_results, args.webhook_url)

    # Exit with appropriate code
    exit_code = 1 if drift_results["overall_drift_detected"] else 0
    return exit_code


if __name__ == "__main__":
    exit(main())
