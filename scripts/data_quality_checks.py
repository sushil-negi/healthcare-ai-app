#!/usr/bin/env python3
"""
Data quality checks for healthcare AI training data
Performs statistical analysis and quality metrics validation
"""

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas/numpy not available, using basic checks")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Performs comprehensive data quality checks"""

    def __init__(self, is_test_file=False):
        self.quality_issues = []
        self.quality_warnings = []
        self.stats = {}
        # Relaxed validation for test files
        self.min_samples_per_category = 1 if is_test_file else 5

    def check_duplicates(self, data: List[Dict]) -> bool:
        """Check for duplicate queries or responses"""
        query_hashes = set()
        response_hashes = set()
        duplicates_found = False

        for i, sample in enumerate(data):
            if "query" not in sample or "response" not in sample:
                continue

            # Hash queries and responses to detect duplicates
            query_hash = hashlib.md5(
                sample["query"].lower().strip().encode()
            ).hexdigest()
            response_hash = hashlib.md5(
                sample["response"].lower().strip().encode()
            ).hexdigest()

            if query_hash in query_hashes:
                self.quality_issues.append(f"Duplicate query found at sample {i}")
                duplicates_found = True
            else:
                query_hashes.add(query_hash)

            if response_hash in response_hashes:
                self.quality_warnings.append(f"Duplicate response found at sample {i}")
            else:
                response_hashes.add(response_hash)

        unique_queries = len(query_hashes)
        unique_responses = len(response_hashes)

        self.stats["unique_queries"] = unique_queries
        self.stats["unique_responses"] = unique_responses
        self.stats["total_samples"] = len(data)

        if not duplicates_found:
            logger.info(f"✅ No duplicate queries found ({unique_queries} unique)")

        if unique_responses == len(data):
            logger.info(f"✅ All responses are unique ({unique_responses})")
        else:
            logger.warning(
                f"⚠️  Some duplicate responses ({unique_responses}/{len(data)} unique)"
            )

        return not duplicates_found

    def check_length_distribution(self, data: List[Dict]) -> bool:
        """Analyze length distribution of queries and responses"""
        query_lengths = []
        response_lengths = []

        for sample in data:
            if "query" in sample:
                query_lengths.append(len(sample["query"]))
            if "response" in sample:
                response_lengths.append(len(sample["response"]))

        if not query_lengths or not response_lengths:
            self.quality_issues.append("No query or response lengths to analyze")
            return False

        # Calculate statistics
        if PANDAS_AVAILABLE:
            query_stats = pd.Series(query_lengths).describe()
            response_stats = pd.Series(response_lengths).describe()

            self.stats["query_length"] = {
                "mean": query_stats["mean"],
                "median": query_stats["50%"],
                "std": query_stats["std"],
                "min": query_stats["min"],
                "max": query_stats["max"],
            }
            self.stats["response_length"] = {
                "mean": response_stats["mean"],
                "median": response_stats["50%"],
                "std": response_stats["std"],
                "min": response_stats["min"],
                "max": response_stats["max"],
            }
        else:
            # Basic statistics without pandas
            self.stats["query_length"] = {
                "mean": sum(query_lengths) / len(query_lengths),
                "min": min(query_lengths),
                "max": max(query_lengths),
            }
            self.stats["response_length"] = {
                "mean": sum(response_lengths) / len(response_lengths),
                "min": min(response_lengths),
                "max": max(response_lengths),
            }

        # Quality checks
        quality_ok = True

        # Check for extremely short queries/responses
        if self.stats["query_length"]["min"] < 5:
            self.quality_issues.append(
                "Some queries are extremely short (< 5 characters)"
            )
            quality_ok = False

        if self.stats["response_length"]["min"] < 10:
            self.quality_issues.append(
                "Some responses are extremely short (< 10 characters)"
            )
            quality_ok = False

        # Check for extremely long queries/responses
        if self.stats["query_length"]["max"] > 500:
            self.quality_warnings.append(
                "Some queries are very long (> 500 characters)"
            )

        if self.stats["response_length"]["max"] > 2000:
            self.quality_warnings.append(
                "Some responses are very long (> 2000 characters)"
            )

        logger.info(f"✅ Length analysis complete:")
        logger.info(
            f"  Query lengths: {self.stats['query_length']['min']}-{self.stats['query_length']['max']} chars (avg: {self.stats['query_length']['mean']:.1f})"
        )
        logger.info(
            f"  Response lengths: {self.stats['response_length']['min']}-{self.stats['response_length']['max']} chars (avg: {self.stats['response_length']['mean']:.1f})"
        )

        return quality_ok

    def check_category_balance(self, data: List[Dict]) -> bool:
        """Check category distribution and balance"""
        category_counts = {}

        for sample in data:
            if "category" in sample:
                category = sample["category"]
                category_counts[category] = category_counts.get(category, 0) + 1

        if not category_counts:
            self.quality_issues.append("No categories found in data")
            return False

        total_samples = sum(category_counts.values())
        self.stats["category_distribution"] = category_counts

        # Check for severe imbalance
        max_count = max(category_counts.values())
        min_count = min(category_counts.values())
        imbalance_ratio = max_count / min_count

        balance_ok = True

        if imbalance_ratio > 10:
            self.quality_issues.append(
                f"Severe category imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
            )
            balance_ok = False
        elif imbalance_ratio > 5:
            self.quality_warnings.append(
                f"Moderate category imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
            )

        # Check for categories with very few samples
        for category, count in category_counts.items():
            percentage = (count / total_samples) * 100
            if count < self.min_samples_per_category:
                self.quality_issues.append(
                    f"Category '{category}' has very few samples: {count} ({percentage:.1f}%)"
                )
                balance_ok = False
            elif percentage < 5:
                self.quality_warnings.append(
                    f"Category '{category}' represents < 5% of data: {count} samples ({percentage:.1f}%)"
                )

        logger.info("✅ Category distribution:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / total_samples) * 100
            logger.info(f"  {category}: {count} samples ({percentage:.1f}%)")

        return balance_ok

    def check_healthcare_specific_quality(self, data: List[Dict]) -> bool:
        """Healthcare-specific quality checks"""
        healthcare_quality_ok = True

        # Check for crisis detection coverage
        crisis_keywords = ["suicide", "kill", "hurt", "die", "end my life", "crisis"]
        crisis_samples = 0

        # Check for medical disclaimer presence
        disclaimer_keywords = [
            "consult",
            "healthcare provider",
            "doctor",
            "professional",
            "medical advice",
        ]
        disclaimer_responses = 0

        # Check for actionable advice
        actionable_keywords = [
            "try",
            "consider",
            "use",
            "contact",
            "call",
            "visit",
            "install",
        ]
        actionable_responses = 0

        for sample in data:
            query = sample.get("query", "").lower()
            response = sample.get("response", "").lower()

            # Crisis detection
            if any(keyword in query for keyword in crisis_keywords):
                crisis_samples += 1
                if "988" not in response:
                    self.quality_warnings.append(
                        "Crisis query missing 988 number in response"
                    )

            # Medical disclaimers
            if any(keyword in response for keyword in disclaimer_keywords):
                disclaimer_responses += 1

            # Actionable advice
            if any(keyword in response for keyword in actionable_keywords):
                actionable_responses += 1

        total_samples = len(data)

        # Quality metrics
        self.stats["healthcare_quality"] = {
            "crisis_samples": crisis_samples,
            "disclaimer_responses": disclaimer_responses,
            "actionable_responses": actionable_responses,
            "crisis_coverage": (crisis_samples / total_samples) * 100,
            "disclaimer_coverage": (disclaimer_responses / total_samples) * 100,
            "actionable_coverage": (actionable_responses / total_samples) * 100,
        }

        # Quality checks
        if crisis_samples == 0:
            self.quality_issues.append("No crisis detection samples found")
            healthcare_quality_ok = False

        if (disclaimer_responses / total_samples) < 0.1:
            self.quality_warnings.append(
                "Less than 10% of responses include medical disclaimers"
            )

        if (actionable_responses / total_samples) < 0.5:
            self.quality_warnings.append(
                "Less than 50% of responses provide actionable advice"
            )

        logger.info("✅ Healthcare-specific quality metrics:")
        logger.info(
            f"  Crisis samples: {crisis_samples} ({self.stats['healthcare_quality']['crisis_coverage']:.1f}%)"
        )
        logger.info(
            f"  Disclaimer responses: {disclaimer_responses} ({self.stats['healthcare_quality']['disclaimer_coverage']:.1f}%)"
        )
        logger.info(
            f"  Actionable responses: {actionable_responses} ({self.stats['healthcare_quality']['actionable_coverage']:.1f}%)"
        )

        return healthcare_quality_ok

    def check_data_quality(self, file_path: str) -> bool:
        """Run comprehensive data quality checks"""
        logger.info("🔍 Starting data quality analysis...")

        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"❌ Data file not found: {file_path}")
            return False

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False

        if not isinstance(data, list):
            logger.error("❌ Data must be a list of samples")
            return False

        logger.info(f"📊 Analyzing {len(data)} samples from {file_path}")

        # Run all quality checks
        checks_passed = [
            self.check_duplicates(data),
            self.check_length_distribution(data),
            self.check_category_balance(data),
            self.check_healthcare_specific_quality(data),
        ]

        all_checks_passed = all(checks_passed)

        # Report results
        if all_checks_passed and not self.quality_issues:
            logger.info("🎉 All data quality checks passed!")
        else:
            if self.quality_issues:
                logger.error("❌ Data quality issues found:")
                for issue in self.quality_issues:
                    logger.error(f"  - {issue}")

            if self.quality_warnings:
                logger.warning("⚠️  Data quality warnings:")
                for warning in self.quality_warnings:
                    logger.warning(f"  - {warning}")

        # Save quality report
        report = {
            "file_path": str(file_path),
            "total_samples": len(data),
            "quality_passed": all_checks_passed and not self.quality_issues,
            "issues": self.quality_issues,
            "warnings": self.quality_warnings,
            "statistics": self.stats,
        }

        report_path = Path("data_quality_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"📝 Quality report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Could not save quality report: {e}")

        return all_checks_passed and not self.quality_issues


def main():
    """Main quality check function"""
    # Default to test data, fall back to others
    data_files = [
        "data/test_healthcare_training.json",
        "data/healthcare_training_data.json",
        "data/combined_healthcare_training_data.json",
    ]

    for data_file in data_files:
        if Path(data_file).exists():
            logger.info(f"Running quality checks on: {data_file}")
            # Use relaxed validation for test files
            is_test_file = "test_" in data_file
            checker = DataQualityChecker(is_test_file=is_test_file)
            success = checker.check_data_quality(data_file)
            sys.exit(0 if success else 1)

    logger.error("❌ No training data files found for quality checks")
    logger.error(f"Looked for: {data_files}")
    sys.exit(1)


if __name__ == "__main__":
    main()
