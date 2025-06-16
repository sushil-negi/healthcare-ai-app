#!/usr/bin/env python3
"""
Create a mock healthcare model for CI testing
"""

import json
import sys
from pathlib import Path

# Create a simple mock model class directly here to avoid import issues
class MockHealthcareModel:
    """Mock healthcare AI model for testing"""
    
    def __init__(self):
        self.model_type = "mock_healthcare_model"
        self.version = "1.0.0"
        self.categories = [
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
            "general_healthcare",
        ]

    def predict(self, X):
        """Mock predict method"""
        if isinstance(X, str):
            X = [X]
        return ["general_healthcare"] * len(X)

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "healthcare_model.joblib"

if JOBLIB_AVAILABLE:
    try:
        # Create and save mock model with joblib
        mock_model = MockHealthcareModel()
        joblib.dump(mock_model, model_path)
        print(f"✅ Mock model created at: {model_path}")
    except Exception as e:
        print(f"❌ Failed to create joblib mock model: {e}")
        JOBLIB_AVAILABLE = False

if not JOBLIB_AVAILABLE:
    # Create a simple JSON fallback
    fallback_model = {
        "model_type": "mock_healthcare_model",
        "version": "1.0.0",
        "categories": [
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
            "general_healthcare",
        ],
    }

    # Save as JSON with .joblib extension for compatibility
    import json

    with open(model_path, "w") as f:
        json.dump(fallback_model, f)
    print(f"✅ Fallback JSON mock model created at: {model_path}")
