#!/usr/bin/env python3
"""
Training data validation for healthcare AI model
Validates data quality, schema, and healthcare-specific requirements
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataValidator:
    """Validates training data for healthcare AI model"""

    def __init__(self, is_test_file=False):
        self.required_fields = ["query", "category", "response"]
        self.valid_categories = [
            "adl_mobility",
            "adl_self_care",
            "senior_medication",
            "senior_social",
            "mental_health_anxiety",
            "mental_health_depression",
            "caregiver_respite",
            "caregiver_burnout",
            "disability_equipment",
            "disability_rights",
            "crisis_mental_health",
        ]
        # Relaxed validation for test files
        self.min_samples_per_category = 1 if is_test_file else 5
        self.validation_errors = []

    def validate_file_exists(self, file_path: Path) -> bool:
        """Check if training data file exists"""
        if not file_path.exists():
            self.validation_errors.append(f"Training data file not found: {file_path}")
            return False

        logger.info(f"✅ Training data file found: {file_path}")
        return True

    def validate_json_format(self, file_path: Path) -> bool:
        """Validate JSON format and structure"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                self.validation_errors.append("Training data must be a list of samples")
                return False

            if len(data) == 0:
                self.validation_errors.append("Training data cannot be empty")
                return False

            logger.info(f"✅ Valid JSON format with {len(data)} samples")
            return True

        except json.JSONDecodeError as e:
            self.validation_errors.append(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.validation_errors.append(f"Error reading file: {e}")
            return False

    def validate_schema(self, data: List[Dict]) -> bool:
        """Validate data schema and required fields"""
        schema_valid = True

        for i, sample in enumerate(data):
            if not isinstance(sample, dict):
                self.validation_errors.append(f"Sample {i} is not a dictionary")
                schema_valid = False
                continue

            # Check required fields
            missing_fields = []
            for field in self.required_fields:
                if field not in sample:
                    missing_fields.append(field)

            if missing_fields:
                self.validation_errors.append(
                    f"Sample {i} missing required fields: {missing_fields}"
                )
                schema_valid = False

            # Validate field types
            if "query" in sample and not isinstance(sample["query"], str):
                self.validation_errors.append(f"Sample {i}: 'query' must be string")
                schema_valid = False

            if "category" in sample and not isinstance(sample["category"], str):
                self.validation_errors.append(f"Sample {i}: 'category' must be string")
                schema_valid = False

            if "response" in sample and not isinstance(sample["response"], str):
                self.validation_errors.append(f"Sample {i}: 'response' must be string")
                schema_valid = False

        if schema_valid:
            logger.info("✅ Schema validation passed")

        return schema_valid

    def validate_categories(self, data: List[Dict]) -> bool:
        """Validate category values and distribution"""
        categories_valid = True
        category_counts = {}

        for i, sample in enumerate(data):
            if "category" not in sample:
                continue

            category = sample["category"]

            # Check if category is valid
            if category not in self.valid_categories:
                self.validation_errors.append(
                    f"Sample {i}: Invalid category '{category}'. "
                    f"Valid categories: {self.valid_categories}"
                )
                categories_valid = False

            # Count categories
            category_counts[category] = category_counts.get(category, 0) + 1

        # Check minimum samples per category
        for category, count in category_counts.items():
            if count < self.min_samples_per_category:
                self.validation_errors.append(
                    f"Category '{category}' has only {count} samples, "
                    f"minimum required: {self.min_samples_per_category}"
                )
                categories_valid = False

        # Check for missing categories
        missing_categories = set(self.valid_categories) - set(category_counts.keys())
        if missing_categories:
            logger.warning(f"⚠️  Missing categories: {missing_categories}")

        if categories_valid:
            logger.info(
                f"✅ Category validation passed. Distribution: {category_counts}"
            )

        return categories_valid

    def validate_content_quality(self, data: List[Dict]) -> bool:
        """Validate content quality of queries and responses"""
        quality_valid = True

        for i, sample in enumerate(data):
            if "query" not in sample or "response" not in sample:
                continue

            query = sample["query"].strip()
            response = sample["response"].strip()

            # Check minimum length
            if len(query) < 10:
                self.validation_errors.append(
                    f"Sample {i}: Query too short ({len(query)} chars), minimum 10"
                )
                quality_valid = False

            if len(response) < 20:
                self.validation_errors.append(
                    f"Sample {i}: Response too short ({len(response)} chars), minimum 20"
                )
                quality_valid = False

            # Check for placeholder text
            placeholders = ["TODO", "PLACEHOLDER", "FILL_IN", "XXX"]
            for placeholder in placeholders:
                if placeholder in query.upper() or placeholder in response.upper():
                    self.validation_errors.append(
                        f"Sample {i}: Contains placeholder text: {placeholder}"
                    )
                    quality_valid = False

        if quality_valid:
            logger.info("✅ Content quality validation passed")

        return quality_valid

    def validate_healthcare_safety(self, data: List[Dict]) -> bool:
        """Validate healthcare-specific safety requirements"""
        safety_valid = True

        crisis_keywords = ["suicide", "kill myself", "hurt myself", "end my life"]
        medical_disclaimers = [
            "consult",
            "healthcare provider",
            "professional",
            "doctor",
        ]

        crisis_responses_found = 0
        medical_disclaimer_responses = 0

        for i, sample in enumerate(data):
            if "query" not in sample or "response" not in sample:
                continue

            query = sample["query"].lower()
            response = sample["response"].lower()

            # Check crisis detection
            if any(keyword in query for keyword in crisis_keywords):
                crisis_responses_found += 1
                if "988" not in response or "crisis" not in response:
                    self.validation_errors.append(
                        f"Sample {i}: Crisis query lacks proper emergency response"
                    )
                    safety_valid = False

            # Check medical disclaimers
            if any(term in response for term in medical_disclaimers):
                medical_disclaimer_responses += 1

        # Ensure we have crisis responses
        if crisis_responses_found == 0:
            self.validation_errors.append(
                "No crisis detection samples found. Healthcare AI must handle emergencies."
            )
            safety_valid = False

        # Ensure medical disclaimers are present
        disclaimer_ratio = medical_disclaimer_responses / len(data)
        if disclaimer_ratio < 0.1:  # At least 10% should have disclaimers
            self.validation_errors.append(
                f"Insufficient medical disclaimers: {disclaimer_ratio:.1%} of responses "
                "(minimum 10% required)"
            )
            safety_valid = False

        if safety_valid:
            logger.info(
                f"✅ Healthcare safety validation passed. "
                f"Crisis responses: {crisis_responses_found}, "
                f"Disclaimer responses: {medical_disclaimer_responses}"
            )

        return safety_valid

    def validate_training_data(self, file_path: str) -> bool:
        """Run complete training data validation"""
        logger.info("🔍 Starting training data validation...")

        file_path = Path(file_path)

        # Validate file exists
        if not self.validate_file_exists(file_path):
            return False

        # Validate JSON format
        if not self.validate_json_format(file_path):
            return False

        # Load data for further validation
        with open(file_path, "r") as f:
            data = json.load(f)

        # Run all validations
        validations = [
            self.validate_schema(data),
            self.validate_categories(data),
            self.validate_content_quality(data),
            self.validate_healthcare_safety(data),
        ]

        all_valid = all(validations)

        if all_valid:
            logger.info("🎉 All training data validations passed!")
        else:
            logger.error("❌ Training data validation failed:")
            for error in self.validation_errors:
                logger.error(f"  - {error}")

        return all_valid


def main():
    """Main validation function"""
    # Default to test data, fall back to others
    data_files = [
        "data/test_healthcare_training.json",
        "data/healthcare_training_data.json",
        "data/combined_healthcare_training_data.json",
    ]

    for data_file in data_files:
        if Path(data_file).exists():
            logger.info(f"Using training data: {data_file}")
            # Use relaxed validation for test files
            is_test_file = "test_" in data_file
            validator = TrainingDataValidator(is_test_file=is_test_file)
            success = validator.validate_training_data(data_file)
            sys.exit(0 if success else 1)

    logger.error("❌ No training data files found")
    logger.error(f"Looked for: {data_files}")
    sys.exit(1)


if __name__ == "__main__":
    main()
