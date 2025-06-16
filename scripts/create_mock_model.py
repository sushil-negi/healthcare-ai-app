#!/usr/bin/env python3
"""
Create a mock healthcare model for CI testing
"""

import os
from pathlib import Path

import joblib


# Mock model class
class MockHealthcareModel:
    def __init__(self):
        self.model_type = "mock_healthcare_model"
        self.version = "1.0.0"
        self.categories = [
            "ADL",
            "Senior Care Services",
            "Mental Health Support",
            "Respite Care",
            "Disability Support",
            "Medication Management",
            "Mobility Assistance",
            "Nutrition Counseling",
            "Social Support",
            "General Healthcare",
            "Crisis/Emergency",
        ]
        self.crisis_keywords = [
            "suicide",
            "kill myself",
            "end it all",
            "hurt myself",
            "die",
        ]

    def predict(self, X):
        """Mock predict method"""
        return ["General Healthcare"] * len(X)

    def predict_proba(self, X):
        """Mock predict_proba method"""
        import numpy as np

        return np.ones((len(X), len(self.categories))) / len(self.categories)


# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Create and save mock model
mock_model = MockHealthcareModel()
model_path = models_dir / "healthcare_model.joblib"

try:
    joblib.dump(mock_model, model_path)
    print(f"✅ Mock model created at: {model_path}")
except Exception as e:
    print(f"❌ Failed to create mock model: {e}")
    # Create a simple dictionary as fallback
    import json

    fallback_model = {"model_type": "mock_healthcare_model", "version": "1.0.0"}
    with open(model_path, "w") as f:
        json.dump(fallback_model, f)
    print(f"✅ Fallback mock model created at: {model_path}")
