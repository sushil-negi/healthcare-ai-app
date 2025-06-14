#!/usr/bin/env python3
"""
HIPAA compliance checks for healthcare AI system
Validates privacy, security, and compliance requirements
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HIPAAComplianceChecker:
    """Performs HIPAA compliance validation"""

    def __init__(self):
        self.compliance_issues = []
        self.compliance_warnings = []
        self.compliance_stats = {}

        # PHI patterns to detect - more specific to avoid false positives
        self.phi_patterns = {
            "names": r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+ [A-Z][a-z]+\b",  # Only with titles
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "dates": r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b",  # More specific date pattern
            "addresses": r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
            "medical_record": r"\b(?:MRN|MR|Patient ID|Record)[\s:]?\d{5,}\b",  # At least 5 digits
            "account_numbers": r"\b(?:Account|Acct)[\s:]?\d{6,}\b",
        }

    def check_phi_exposure(self, data: List[Dict]) -> bool:
        """Check for potential PHI exposure in training data"""
        phi_found = False
        phi_counts = {pattern_name: 0 for pattern_name in self.phi_patterns}

        for i, sample in enumerate(data):
            query = sample.get("query", "")
            response = sample.get("response", "")

            # Check both query and response for PHI patterns
            for text_type, text in [("query", query), ("response", response)]:
                for pattern_name, pattern in self.phi_patterns.items():
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        phi_counts[pattern_name] += len(matches)
                        self.compliance_issues.append(
                            f"Sample {i} {text_type} contains potential {pattern_name}: {matches[0]}"
                        )
                        phi_found = True

        self.compliance_stats["phi_patterns_found"] = phi_counts
        total_phi = sum(phi_counts.values())

        if phi_found:
            logger.error(f"❌ PHI exposure detected: {total_phi} potential instances")
            for pattern_name, count in phi_counts.items():
                if count > 0:
                    logger.error(f"  {pattern_name}: {count} instances")
        else:
            logger.info("✅ No obvious PHI patterns detected")

        return not phi_found

    def check_medical_advice_disclaimers(self, data: List[Dict]) -> bool:
        """Ensure appropriate medical disclaimers are present"""
        disclaimer_keywords = [
            "consult",
            "healthcare provider",
            "doctor",
            "physician",
            "medical professional",
            "healthcare professional",
            "medical advice",
            "qualified healthcare",
        ]

        responses_with_disclaimers = 0
        medical_query_keywords = [
            "medication",
            "symptoms",
            "treatment",
            "diagnosis",
            "medical",
            "health",
            "pain",
            "illness",
            "disease",
            "therapy",
            "prescription",
        ]

        medical_queries = 0
        medical_queries_with_disclaimers = 0

        for sample in data:
            query = sample.get("query", "").lower()
            response = sample.get("response", "").lower()

            # Check if query is medical in nature
            is_medical_query = any(
                keyword in query for keyword in medical_query_keywords
            )
            if is_medical_query:
                medical_queries += 1

                # Check if response has disclaimer
                has_disclaimer = any(
                    keyword in response for keyword in disclaimer_keywords
                )
                if has_disclaimer:
                    medical_queries_with_disclaimers += 1
                    responses_with_disclaimers += 1
                else:
                    self.compliance_warnings.append(
                        f"Medical query lacks healthcare disclaimer: '{query[:50]}...'"
                    )
            elif any(keyword in response for keyword in disclaimer_keywords):
                responses_with_disclaimers += 1

        total_samples = len(data)
        disclaimer_coverage = (
            (responses_with_disclaimers / total_samples) * 100
            if total_samples > 0
            else 0
        )
        medical_disclaimer_coverage = (
            (medical_queries_with_disclaimers / medical_queries) * 100
            if medical_queries > 0
            else 0
        )

        self.compliance_stats["disclaimer_coverage"] = {
            "total_responses_with_disclaimers": responses_with_disclaimers,
            "medical_queries": medical_queries,
            "medical_queries_with_disclaimers": medical_queries_with_disclaimers,
            "overall_coverage": disclaimer_coverage,
            "medical_coverage": medical_disclaimer_coverage,
        }

        compliance_ok = True

        # Requirements: Medical queries should have disclaimers
        if medical_queries > 0 and medical_disclaimer_coverage < 80:
            self.compliance_issues.append(
                f"Insufficient medical disclaimers: {medical_disclaimer_coverage:.1f}% "
                f"of medical queries have disclaimers (minimum 80% required)"
            )
            compliance_ok = False

        logger.info(f"✅ Medical disclaimer analysis:")
        logger.info(f"  Medical queries: {medical_queries}")
        logger.info(
            f"  Medical queries with disclaimers: {medical_queries_with_disclaimers} ({medical_disclaimer_coverage:.1f}%)"
        )
        logger.info(f"  Overall disclaimer coverage: {disclaimer_coverage:.1f}%")

        return compliance_ok

    def check_crisis_response_compliance(self, data: List[Dict]) -> bool:
        """Ensure crisis responses meet compliance standards"""
        crisis_keywords = [
            "suicide",
            "kill myself",
            "kill me",
            "hurt myself",
            "hurt me",
            "end my life",
            "end everything",
            "self-harm",
            "self harm",
            "don't want to live",
            "want to die",
            "better off dead",
        ]
        emergency_resources = ["988", "911", "crisis", "emergency", "hotline"]

        crisis_queries = 0
        compliant_crisis_responses = 0

        for sample in data:
            query = sample.get("query", "").lower()
            response = sample.get("response", "").lower()

            # Check if query indicates crisis
            is_crisis_query = any(keyword in query for keyword in crisis_keywords)
            if is_crisis_query:
                crisis_queries += 1

                # Check if response provides emergency resources
                has_emergency_resources = any(
                    resource in response for resource in emergency_resources
                )
                if has_emergency_resources:
                    compliant_crisis_responses += 1
                else:
                    self.compliance_issues.append(
                        f"Crisis query lacks emergency resources: '{query[:50]}...'"
                    )

        compliance_ok = True

        if crisis_queries > 0:
            crisis_compliance_rate = (compliant_crisis_responses / crisis_queries) * 100

            if crisis_compliance_rate < 100:
                self.compliance_issues.append(
                    f"Crisis response compliance: {crisis_compliance_rate:.1f}% "
                    f"(must be 100% - all crisis queries must provide emergency resources)"
                )
                compliance_ok = False

        self.compliance_stats["crisis_compliance"] = {
            "crisis_queries": crisis_queries,
            "compliant_responses": compliant_crisis_responses,
            "compliance_rate": (
                (compliant_crisis_responses / crisis_queries * 100)
                if crisis_queries > 0
                else 0
            ),
        }

        logger.info(f"✅ Crisis response compliance:")
        logger.info(f"  Crisis queries: {crisis_queries}")
        logger.info(f"  Compliant responses: {compliant_crisis_responses}")
        if crisis_queries > 0:
            logger.info(
                f"  Compliance rate: {(compliant_crisis_responses / crisis_queries * 100):.1f}%"
            )

        return compliance_ok

    def check_data_minimization(self, data: List[Dict]) -> bool:
        """Check data minimization principles"""
        # Analyze if we're collecting only necessary information
        field_usage = {}

        for sample in data:
            for field in sample.keys():
                field_usage[field] = field_usage.get(field, 0) + 1

        total_samples = len(data)
        required_fields = ["query", "category", "response"]
        unnecessary_fields = []

        for field, count in field_usage.items():
            usage_rate = (count / total_samples) * 100

            if field not in required_fields:
                if usage_rate < 50:  # Field used in less than 50% of samples
                    unnecessary_fields.append(f"{field} ({usage_rate:.1f}% usage)")
                else:
                    self.compliance_warnings.append(
                        f"Additional field '{field}' present in {usage_rate:.1f}% of samples"
                    )

        self.compliance_stats["data_minimization"] = {
            "field_usage": field_usage,
            "unnecessary_fields": unnecessary_fields,
        }

        compliance_ok = True

        if unnecessary_fields:
            self.compliance_warnings.append(
                f"Potential unnecessary fields found: {', '.join(unnecessary_fields)}"
            )

        logger.info(f"✅ Data minimization check:")
        logger.info(f"  Fields found: {list(field_usage.keys())}")
        logger.info(f"  Required fields: {required_fields}")
        if unnecessary_fields:
            logger.info(f"  Low-usage fields: {unnecessary_fields}")

        return compliance_ok

    def check_access_controls(self) -> bool:
        """Check if proper access controls are documented"""
        # This is a placeholder for access control verification
        # In a real implementation, this would check:
        # - Authentication requirements
        # - Authorization policies
        # - Audit logging
        # - Data encryption

        access_control_requirements = [
            "Authentication required for all data access",
            "Role-based access controls implemented",
            "Audit logging for all data operations",
            "Data encryption at rest and in transit",
            "Regular access reviews conducted",
        ]

        self.compliance_stats["access_controls"] = {
            "requirements": access_control_requirements,
            "status": "Documentation review required",
        }

        self.compliance_warnings.append(
            "Access control compliance requires manual review of system documentation"
        )

        logger.info("✅ Access control requirements documented")
        logger.info("  Manual review required for full compliance verification")

        return True

    def run_hipaa_compliance_check(self, file_path: str) -> bool:
        """Run comprehensive HIPAA compliance checks"""
        logger.info("🔒 Starting HIPAA compliance validation...")

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

        logger.info(f"🔍 Analyzing {len(data)} samples for HIPAA compliance...")

        # Run all compliance checks
        checks_passed = [
            self.check_phi_exposure(data),
            self.check_medical_advice_disclaimers(data),
            self.check_crisis_response_compliance(data),
            self.check_data_minimization(data),
            self.check_access_controls(),
        ]

        all_checks_passed = all(checks_passed)

        # Report results
        if all_checks_passed and not self.compliance_issues:
            logger.info("🎉 All HIPAA compliance checks passed!")
        else:
            if self.compliance_issues:
                logger.error("❌ HIPAA compliance issues found:")
                for issue in self.compliance_issues:
                    logger.error(f"  - {issue}")

            if self.compliance_warnings:
                logger.warning("⚠️  HIPAA compliance warnings:")
                for warning in self.compliance_warnings:
                    logger.warning(f"  - {warning}")

        # Save compliance report
        report = {
            "file_path": str(file_path),
            "total_samples": len(data),
            "compliance_passed": all_checks_passed and not self.compliance_issues,
            "issues": self.compliance_issues,
            "warnings": self.compliance_warnings,
            "statistics": self.compliance_stats,
        }

        report_path = Path("hipaa_compliance_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"📝 Compliance report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Could not save compliance report: {e}")

        return all_checks_passed and not self.compliance_issues


def main():
    """Main HIPAA compliance check function"""
    # Default to test data, fall back to others
    data_files = [
        "data/test_healthcare_training.json",
        "data/healthcare_training_data.json",
        "data/combined_healthcare_training_data.json",
    ]

    checker = HIPAAComplianceChecker()

    for data_file in data_files:
        if Path(data_file).exists():
            logger.info(f"Running HIPAA compliance checks on: {data_file}")
            success = checker.run_hipaa_compliance_check(data_file)
            sys.exit(0 if success else 1)

    logger.error("❌ No training data files found for HIPAA compliance checks")
    logger.error(f"Looked for: {data_files}")
    sys.exit(1)


if __name__ == "__main__":
    main()
