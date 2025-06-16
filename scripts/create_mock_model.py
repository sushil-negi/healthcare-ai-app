#!/usr/bin/env python3
"""
Create a mock healthcare model for CI testing
"""

import sys
from pathlib import Path

# Add scripts directory to path to import mock model
sys.path.insert(0, str(Path(__file__).parent))

try:
    import joblib
    from healthcare_mock_model import MockHealthcareModel

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    import json


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
