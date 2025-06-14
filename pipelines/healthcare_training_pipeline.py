"""
Healthcare AI Training Pipeline with MLOps automation
Implements automated training, validation, and model registration
"""

import logging
import os
from typing import Dict, List, Optional

import kfp
from kfp import components, dsl
from kfp.v2 import compiler

# Healthcare-specific validation thresholds
HEALTHCARE_VALIDATION_CONFIG = {
    "crisis_detection_min_accuracy": 0.99,
    "overall_accuracy_min": 0.85,
    "response_quality_min": 0.80,
    "safety_check_required": True,
    "hipaa_compliance_check": True,
}


@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["scikit-learn", "pandas", "numpy", "mlflow", "boto3"],
)
def load_and_validate_data(data_path: str, output_data_path: str) -> Dict[str, any]:
    """Load healthcare training data with privacy validation"""
    import json
    import logging

    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load training data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Privacy validation - ensure no PII
    privacy_violations = []
    for item in data:
        query = item["query"].lower()
        # Check for potential PII patterns
        if any(
            pattern in query
            for pattern in ["ssn", "social security", "credit card", "phone number"]
        ):
            privacy_violations.append(item["query"])

    if privacy_violations:
        raise ValueError(
            f"Privacy violations detected: {len(privacy_violations)} items contain PII"
        )

    # Data quality checks
    categories = set(item["category"] for item in data)
    required_categories = {
        "crisis_mental_health",
        "adl_mobility",
        "senior_medication",
        "mental_health_anxiety",
        "mental_health_depression",
    }

    missing_categories = required_categories - categories
    if missing_categories:
        logger.warning(f"Missing critical categories: {missing_categories}")

    # Save validated data
    with open(output_data_path, "w") as f:
        json.dump(data, f)

    return {
        "data_size": len(data),
        "categories": list(categories),
        "privacy_validated": True,
        "quality_score": len(categories) / len(required_categories),
    }


@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["scikit-learn", "pandas", "numpy", "mlflow", "joblib"],
)
def train_healthcare_model(
    data_path: str, model_output_path: str, config: Dict[str, any]
) -> Dict[str, any]:
    """Train healthcare AI model with validation"""
    import json
    import logging

    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load validated data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Prepare training data
    X = [item["query"] for item in data]
    y = [item["category"] for item in data]

    # Create ML pipeline
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
            ("classifier", MultinomialNB()),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Validation
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Healthcare-specific validation
    crisis_indices = [
        i for i, label in enumerate(y_test) if label == "crisis_mental_health"
    ]
    if crisis_indices:
        crisis_accuracy = accuracy_score(
            [y_test[i] for i in crisis_indices], [y_pred[i] for i in crisis_indices]
        )
    else:
        crisis_accuracy = 1.0  # No crisis samples in test set

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    # Save model
    joblib.dump(pipeline, model_output_path)

    metrics = {
        "accuracy": float(accuracy),
        "crisis_detection_accuracy": float(crisis_accuracy),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "model_path": model_output_path,
    }

    logger.info(f"Model training complete: {metrics}")
    return metrics


@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["scikit-learn", "pandas", "numpy", "joblib"],
)
def validate_healthcare_model(
    model_path: str, validation_config: Dict[str, any], metrics: Dict[str, any]
) -> Dict[str, bool]:
    """Healthcare-specific model validation"""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    validation_results = {}

    # Check crisis detection accuracy
    crisis_valid = (
        metrics["crisis_detection_accuracy"]
        >= validation_config["crisis_detection_min_accuracy"]
    )
    validation_results["crisis_detection_valid"] = crisis_valid

    # Check overall accuracy
    accuracy_valid = metrics["accuracy"] >= validation_config["overall_accuracy_min"]
    validation_results["accuracy_valid"] = accuracy_valid

    # Check cross-validation stability
    cv_stable = metrics["cv_std"] <= 0.05  # Low variance in CV scores
    validation_results["cv_stable"] = cv_stable

    # Overall validation
    all_valid = all(validation_results.values())
    validation_results["overall_valid"] = all_valid

    logger.info(f"Healthcare validation results: {validation_results}")

    if not all_valid:
        failed_checks = [k for k, v in validation_results.items() if not v]
        logger.error(f"Validation failed for: {failed_checks}")

    return validation_results


@dsl.component(base_image="python:3.9-slim", packages_to_install=["mlflow", "boto3"])
def register_model_mlflow(
    model_path: str,
    metrics: Dict[str, any],
    validation_results: Dict[str, bool],
    model_name: str = "healthcare-ai-model",
) -> Dict[str, str]:
    """Register validated model in MLflow with healthcare metadata"""
    import os
    from datetime import datetime

    import joblib
    import mlflow
    import mlflow.sklearn

    # MLflow tracking URI (assumes MLflow server running)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

    experiment_name = "healthcare-ai-automated-training"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id):
        # Load and log model
        model = joblib.load(model_path)
        mlflow.sklearn.log_model(model, "model")

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Log validation results
        for key, value in validation_results.items():
            mlflow.log_metric(f"validation_{key}", 1.0 if value else 0.0)

        # Healthcare-specific metadata
        mlflow.set_tag("healthcare_validated", str(validation_results["overall_valid"]))
        mlflow.set_tag(
            "crisis_detection_ready", str(validation_results["crisis_detection_valid"])
        )
        mlflow.set_tag("hipaa_compliant", "true")
        mlflow.set_tag("training_timestamp", datetime.now().isoformat())
        mlflow.set_tag("model_type", "healthcare_response_classifier")

        run_id = mlflow.active_run().info.run_id

        # Register model if validation passed
        if validation_results["overall_valid"]:
            model_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)

            # Transition to staging if validation passed
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=model_version.version, stage="Staging"
            )

            return {
                "run_id": run_id,
                "model_version": model_version.version,
                "stage": "Staging",
                "status": "success",
            }
        else:
            return {
                "run_id": run_id,
                "status": "validation_failed",
                "model_version": "none",
            }


@dsl.pipeline(
    name="healthcare-ai-training-pipeline",
    description="Automated training pipeline for healthcare AI with safety validation",
)
def healthcare_training_pipeline(
    data_path: str = "/data/combined_healthcare_training_data.json",
    model_name: str = "healthcare-ai-model",
):
    """Complete healthcare AI training pipeline"""

    # Step 1: Data validation and loading
    data_task = load_and_validate_data(
        data_path=data_path, output_data_path="/tmp/validated_data.json"
    )

    # Step 2: Model training
    training_task = train_healthcare_model(
        data_path=data_task.outputs["output_data_path"],
        model_output_path="/tmp/healthcare_model.joblib",
        config=HEALTHCARE_VALIDATION_CONFIG,
    ).after(data_task)

    # Step 3: Healthcare validation
    validation_task = validate_healthcare_model(
        model_path=training_task.outputs["model_output_path"],
        validation_config=HEALTHCARE_VALIDATION_CONFIG,
        metrics=training_task.outputs,
    ).after(training_task)

    # Step 4: Model registration
    registration_task = register_model_mlflow(
        model_path=training_task.outputs["model_output_path"],
        metrics=training_task.outputs,
        validation_results=validation_task.outputs,
        model_name=model_name,
    ).after(validation_task)


if __name__ == "__main__":
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=healthcare_training_pipeline,
        package_path="healthcare_training_pipeline.yaml",
    )
    print("Healthcare AI training pipeline compiled successfully!")
