#!/usr/bin/env python3
"""
Response quality validation for healthcare AI
Validates response quality metrics including clarity, completeness, and safety
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseQualityValidator:
    """Validates response quality for healthcare AI"""

    def __init__(self):
        self.validation_results = {}
        self.test_data = None

    def load_test_data(self) -> bool:
        """Load test data for validation"""
        test_data_paths = [
            "data/test_healthcare_training.json",
            "data/combined_healthcare_training_data.json",
            "data/healthcare_training_data.json",
        ]

        for data_path in test_data_paths:
            if Path(data_path).exists():
                try:
                    with open(data_path, "r") as f:
                        data = json.load(f)

                    if isinstance(data, list) and len(data) > 0:
                        self.test_data = data
                        logger.info(
                            f"✅ Test data loaded: {len(data)} samples from {data_path}"
                        )
                        return True
                except Exception as e:
                    logger.warning(f"Failed to load test data from {data_path}: {e}")

        logger.error("❌ No test data found")
        return False

    def validate_response_clarity(self) -> bool:
        """Validate response clarity and structure"""
        if not self.test_data:
            return False

        logger.info("📝 Validating response clarity...")

        clarity_issues = []
        total_responses = 0
        clear_responses = 0

        # Clarity indicators
        clarity_indicators = [
            "•",  # Bullet points
            "\n",  # Proper formatting
            ":",  # Clear sections
        ]

        unclear_patterns = [
            "...",  # Excessive ellipsis
            "???",  # Confusion
            "!!!!!",  # Excessive exclamation
        ]

        for i, sample in enumerate(self.test_data):
            if "response" not in sample:
                continue

            response = sample["response"]
            total_responses += 1

            # Check for clarity indicators
            has_structure = any(
                indicator in response for indicator in clarity_indicators
            )
            has_unclear = any(pattern in response for pattern in unclear_patterns)

            if has_structure and not has_unclear:
                clear_responses += 1
            else:
                clarity_issues.append(f"Sample {i}: Response lacks clear structure")

        clarity_rate = clear_responses / total_responses if total_responses > 0 else 0

        self.validation_results["clarity"] = {
            "total_responses": total_responses,
            "clear_responses": clear_responses,
            "clarity_rate": clarity_rate,
            "issues": len(clarity_issues),
        }

        required_clarity = 0.8  # 80% clarity required
        if clarity_rate >= required_clarity:
            logger.info(
                f"✅ Response clarity: {clarity_rate:.1%} (meets requirement: {required_clarity:.1%})"
            )
            return True
        else:
            logger.error(
                f"❌ Response clarity: {clarity_rate:.1%} (required: {required_clarity:.1%})"
            )
            return False

    def validate_response_completeness(self) -> bool:
        """Validate response completeness"""
        if not self.test_data:
            return False

        logger.info("🔍 Validating response completeness...")

        completeness_metrics = {
            "has_advice": 0,
            "has_disclaimer": 0,
            "has_resources": 0,
            "has_action_items": 0,
            "total": 0,
        }

        for sample in self.test_data:
            if "response" not in sample:
                continue

            response = sample["response"].lower()
            completeness_metrics["total"] += 1

            # Check components
            if any(
                keyword in response
                for keyword in ["try", "consider", "use", "contact", "consult"]
            ):
                completeness_metrics["has_advice"] += 1

            if any(
                keyword in response
                for keyword in ["healthcare provider", "professional", "doctor"]
            ):
                completeness_metrics["has_disclaimer"] += 1

            if any(
                keyword in response
                for keyword in ["resources", "support", "services", "help"]
            ):
                completeness_metrics["has_resources"] += 1

            if "•" in sample["response"] or any(
                action in response for action in ["step", "first", "next", "then"]
            ):
                completeness_metrics["has_action_items"] += 1

        # Calculate completeness scores
        total = completeness_metrics["total"]
        if total == 0:
            return False

        completeness_scores = {
            "advice_rate": completeness_metrics["has_advice"] / total,
            "disclaimer_rate": completeness_metrics["has_disclaimer"] / total,
            "resources_rate": completeness_metrics["has_resources"] / total,
            "action_items_rate": completeness_metrics["has_action_items"] / total,
        }

        self.validation_results["completeness"] = {
            "metrics": completeness_metrics,
            "scores": completeness_scores,
        }

        # All components should be present in majority of responses
        min_completeness = 0.7
        all_complete = all(
            score >= min_completeness for score in completeness_scores.values()
        )

        if all_complete:
            logger.info("✅ Response completeness validation passed")
            for component, score in completeness_scores.items():
                logger.info(f"  {component}: {score:.1%}")
        else:
            logger.error("❌ Response completeness validation failed")
            for component, score in completeness_scores.items():
                if score < min_completeness:
                    logger.error(
                        f"  {component}: {score:.1%} (minimum {min_completeness:.1%})"
                    )

        return all_complete

    def validate_response_safety(self) -> bool:
        """Validate response safety"""
        if not self.test_data:
            return False

        logger.info("🛡️ Validating response safety...")

        safety_issues = []
        total_medical_responses = 0
        safe_medical_responses = 0

        # Unsafe patterns
        unsafe_patterns = [
            "definitely",
            "always",
            "never",
            "guaranteed",
            "cure",
            "100%",
            "proven to",
        ]

        # Safe patterns
        safe_patterns = [
            "may",
            "might",
            "could",
            "consider",
            "consult",
            "professional",
            "healthcare provider",
        ]

        for i, sample in enumerate(self.test_data):
            if "response" not in sample:
                continue

            response = sample["response"].lower()
            category = sample.get("category", "")

            # Focus on medical categories
            if any(
                med in category
                for med in ["medication", "health", "medical", "treatment"]
            ):
                total_medical_responses += 1

                # Check for unsafe absolute statements
                has_unsafe = any(pattern in response for pattern in unsafe_patterns)
                has_safe = any(pattern in response for pattern in safe_patterns)

                if not has_unsafe and has_safe:
                    safe_medical_responses += 1
                else:
                    safety_issues.append(
                        f"Sample {i}: Medical response may contain unsafe absolute statements"
                    )

        safety_rate = (
            safe_medical_responses / total_medical_responses
            if total_medical_responses > 0
            else 1.0
        )

        self.validation_results["safety"] = {
            "total_medical_responses": total_medical_responses,
            "safe_medical_responses": safe_medical_responses,
            "safety_rate": safety_rate,
            "issues": len(safety_issues),
        }

        required_safety = 0.9  # 90% safety required
        if safety_rate >= required_safety:
            logger.info(
                f"✅ Response safety: {safety_rate:.1%} (meets requirement: {required_safety:.1%})"
            )
            return True
        else:
            logger.error(
                f"❌ Response safety: {safety_rate:.1%} (required: {required_safety:.1%})"
            )
            return False

    def validate_response_empathy(self) -> bool:
        """Validate response empathy and tone"""
        if not self.test_data:
            return False

        logger.info("💚 Validating response empathy...")

        empathy_metrics = {
            "total": 0,
            "empathetic": 0,
            "supportive": 0,
            "personalized": 0,
        }

        # Empathy indicators
        empathy_words = [
            "understand",
            "support",
            "help",
            "care",
            "matter",
            "available",
            "here for",
            "reach out",
        ]

        supportive_phrases = [
            "you're not alone",
            "help is available",
            "support available",
            "we understand",
            "here to help",
        ]

        for sample in self.test_data:
            if "response" not in sample:
                continue

            response = sample["response"].lower()
            empathy_metrics["total"] += 1

            # Check empathy indicators
            if any(word in response for word in empathy_words):
                empathy_metrics["empathetic"] += 1

            if any(phrase in response for phrase in supportive_phrases):
                empathy_metrics["supportive"] += 1

            # Check for personalization (use of "you", "your")
            if response.count("you") >= 2:
                empathy_metrics["personalized"] += 1

        # Calculate empathy scores
        total = empathy_metrics["total"]
        if total == 0:
            return False

        empathy_scores = {
            "empathy_rate": empathy_metrics["empathetic"] / total,
            "support_rate": empathy_metrics["supportive"] / total,
            "personalization_rate": empathy_metrics["personalized"] / total,
        }

        self.validation_results["empathy"] = {
            "metrics": empathy_metrics,
            "scores": empathy_scores,
        }

        # Average empathy score
        avg_empathy = sum(empathy_scores.values()) / len(empathy_scores)
        required_empathy = 0.65  # 65% empathy required

        if avg_empathy >= required_empathy:
            logger.info(
                f"✅ Response empathy: {avg_empathy:.1%} (meets requirement: {required_empathy:.1%})"
            )
            return True
        else:
            logger.error(
                f"❌ Response empathy: {avg_empathy:.1%} (required: {required_empathy:.1%})"
            )
            return False

    def run_response_quality_validation(self) -> bool:
        """Run complete response quality validation"""
        logger.info("📊 Starting response quality validation...")

        # Load test data
        if not self.load_test_data():
            return False

        # Run all validations
        validations = [
            self.validate_response_clarity(),
            self.validate_response_completeness(),
            self.validate_response_safety(),
            self.validate_response_empathy(),
        ]

        all_passed = all(validations)

        # Save validation report
        report = {
            "validation_passed": all_passed,
            "timestamp": time.time(),
            "results": self.validation_results,
        }

        try:
            with open("response_quality_validation_report.json", "w") as f:
                json.dump(report, f, indent=2)
            logger.info(
                "📝 Validation report saved to response_quality_validation_report.json"
            )
        except Exception as e:
            logger.warning(f"Could not save validation report: {e}")

        if all_passed:
            logger.info("🎉 All response quality validations passed!")
        else:
            logger.error("❌ Some response quality validations failed")

        return all_passed


def main():
    """Main response quality validation function"""
    validator = ResponseQualityValidator()
    success = validator.run_response_quality_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
