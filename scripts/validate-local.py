#!/usr/bin/env python3
"""
Local CI/CD Validation Script for Healthcare AI App
Runs the same checks as the CI/CD pipeline
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔍 {description}...")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        print(f"✅ {description} passed ({duration:.2f}s)")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ {description} failed ({duration:.2f}s)")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def main():
    """Run healthcare AI validation checks"""
    print("🏥 Healthcare AI App - Local CI/CD Validation")
    print("=" * 50)

    # Change to project root
    project_root = Path(__file__).parent.parent
    original_dir = Path.cwd()

    try:
        import os

        os.chdir(project_root)

        checks = []

        # Code formatting
        checks.append(
            run_command(
                [
                    "python3",
                    "-m",
                    "black",
                    "--check",
                    "--diff",
                    "src/",
                    "tests/",
                    "scripts/",
                    "monitoring/",
                ],
                "Black code formatting",
            )
        )

        # Import sorting
        checks.append(
            run_command(
                [
                    "python3",
                    "-m",
                    "isort",
                    "--check-only",
                    "--diff",
                    "src/",
                    "tests/",
                    "scripts/",
                    "monitoring/",
                ],
                "Import sorting (isort)",
            )
        )

        # Linting
        checks.append(
            run_command(
                [
                    "python3",
                    "-m",
                    "flake8",
                    "--max-line-length=88",
                    "--extend-ignore=E203,W503",
                    "src/",
                    "tests/",
                    "scripts/",
                    "monitoring/",
                ],
                "Flake8 linting",
            )
        )

        # Type checking
        checks.append(
            run_command(
                [
                    "python3",
                    "-m",
                    "mypy",
                    "--ignore-missing-imports",
                    "--no-strict-optional",
                    "src/",
                ],
                "MyPy type checking",
            )
        )

        # Security scanning
        checks.append(
            run_command(
                ["python3", "-m", "bandit", "-r", "-ll", "src/", "scripts/"],
                "Bandit security scan",
            )
        )

        # Unit tests
        checks.append(
            run_command(["python3", "-m", "pytest", "tests/unit/", "-v"], "Unit tests")
        )

        # Integration tests
        checks.append(
            run_command(
                ["python3", "-m", "pytest", "tests/integration/", "-v"],
                "Integration tests",
            )
        )

        # Healthcare-specific validations
        if (project_root / "scripts" / "hipaa_compliance_check.py").exists():
            checks.append(
                run_command(
                    ["python3", "scripts/hipaa_compliance_check.py"],
                    "HIPAA compliance check",
                )
            )

        if (project_root / "scripts" / "validate_training_data.py").exists():
            checks.append(
                run_command(
                    ["python3", "scripts/validate_training_data.py"],
                    "Training data validation",
                )
            )

        # Summary
        passed = sum(checks)
        total = len(checks)

        print("\n" + "=" * 50)
        print(f"🎯 VALIDATION SUMMARY")
        print(f"   Passed: {passed}/{total}")

        if passed == total:
            print("🎉 All validations passed - ready to commit!")
            return True
        else:
            print(f"❌ {total - passed} validation(s) failed")
            return False

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
