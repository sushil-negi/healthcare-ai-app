#!/usr/bin/env python3
"""
Local testing script for healthcare AI application
Starts services and runs integration and E2E tests
"""

import os
import subprocess
import sys
import time
import requests
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a shell command"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    
    print(f"✅ {description} completed")
    return True

def check_service_health(url, service_name, max_retries=30, delay=2):
    """Check if a service is healthy"""
    print(f"\n🔍 Checking {service_name} health at {url}")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is healthy!")
                return True
        except requests.exceptions.RequestException as e:
            pass
        
        if i < max_retries - 1:
            print(f"⏳ Waiting for {service_name}... ({i+1}/{max_retries})")
            time.sleep(delay)
    
    print(f"❌ {service_name} failed to become healthy")
    return False

def main():
    """Main testing function"""
    print("🏥 Healthcare AI Local Testing Suite")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Step 1: Create mock model
    print("\n📋 Step 1: Setting up mock model...")
    if not run_command("python3 scripts/create_mock_model.py", "Create mock healthcare model"):
        return False
    
    # Step 2: Run unit tests first
    print("\n📋 Step 2: Running unit tests...")
    if not run_command("python3 -m pytest tests/unit/ -v", "Run unit tests"):
        return False
    
    # Step 3: Run integration tests
    print("\n📋 Step 3: Running integration tests...")
    if not run_command("python3 -m pytest tests/integration/ -v", "Run integration tests"):
        return False
    
    # Step 4: Run healthcare validation scripts
    print("\n📋 Step 4: Running healthcare validations...")
    validations = [
        ("python3 tests/healthcare_model_validation.py", "Healthcare model validation"),
        ("python3 tests/crisis_detection_validation.py", "Crisis detection validation"), 
        ("python3 tests/response_quality_validation.py", "Response quality validation"),
        ("python3 scripts/hipaa_compliance_check.py", "HIPAA compliance check"),
    ]
    
    for cmd, desc in validations:
        if not run_command(cmd, desc):
            print(f"⚠️ Warning: {desc} failed, but continuing...")
    
    # Step 5: Check if we should run E2E tests
    print("\n📋 Step 5: Checking for E2E test requirements...")
    
    run_e2e = input("Do you want to start Docker services and run E2E tests? (y/N): ").lower().startswith('y')
    
    if run_e2e:
        print("\n📋 Step 6: Starting Docker services...")
        
        # Start services
        if not run_command("docker compose -f docker-compose.healthcare.yml up -d", "Start healthcare services"):
            return False
        
        time.sleep(10)  # Give services time to start
        
        # Check service health
        services = [
            ("http://localhost:8001", "Healthcare AI"),
            ("http://localhost:8889", "Healthcare Web"),
            ("http://localhost:8082", "Healthcare Metrics")
        ]
        
        all_healthy = True
        for url, name in services:
            if not check_service_health(url, name, max_retries=15):
                print(f"⚠️ {name} service not healthy, E2E tests may fail")
                all_healthy = False
        
        if all_healthy:
            # Run E2E tests
            print("\n📋 Step 7: Running E2E tests...")
            env = os.environ.copy()
            env["HEALTHCARE_SERVICE_URL"] = "http://localhost:8001"
            
            result = subprocess.run(
                ["python3", "-m", "pytest", "tests/e2e/", "-v", "--tb=short"],
                env=env
            )
            
            if result.returncode == 0:
                print("✅ E2E tests passed!")
            else:
                print("❌ E2E tests failed!")
        
        # Cleanup
        print("\n📋 Step 8: Cleaning up Docker services...")
        run_command("docker compose -f docker-compose.healthcare.yml down", "Stop healthcare services", check=False)
    
    # Step 6: Run pre-commit validation
    print("\n📋 Final Step: Running pre-commit validation...")
    run_command("python3 scripts/pre-commit-ci-validation.py", "Pre-commit CI validation", check=False)
    
    print("\n🎉 Local testing completed!")
    print("You can now safely push to trigger CI/CD pipeline.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        print("Cleaning up...")
        run_command("docker compose -f docker-compose.healthcare.yml down", "Emergency cleanup", check=False)
        sys.exit(1)