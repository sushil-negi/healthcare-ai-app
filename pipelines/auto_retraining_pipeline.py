#!/usr/bin/env python3
"""
Healthcare AI Auto-Retraining Pipeline
Automatically retrains models with new conversation data
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

# MLOps Platform Services
MODEL_REGISTRY_URL = "http://localhost:8000"
EXPERIMENT_TRACKING_URL = "http://localhost:8003"
PIPELINE_ORCHESTRATOR_URL = "http://localhost:8004"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoRetrainingPipeline:
    def __init__(self):
        self.pipeline_id = f"auto-retrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.data_dir = Path("/app/data/retraining")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Retraining configuration
        self.config = {
            "min_new_conversations": 1000,  # Minimum new conversations before retraining
            "performance_degradation_threshold": 0.05,  # 5% performance drop triggers retraining
            "retraining_interval_days": 7,  # Maximum days between retraining
            "data_quality_threshold": 0.95,  # Minimum data quality score
            "auto_deploy_threshold": 0.02,  # Auto-deploy if improvement > 2%
        }

    def collect_new_conversations(self) -> List[Dict]:
        """Collect new conversations from production"""
        logger.info("📥 Collecting new conversations from production...")

        # In production, this would query the healthcare app database
        # For demo, we'll simulate with synthetic data
        new_conversations = []

        try:
            # Check for conversation logs
            conversation_log = self.data_dir / "conversation_logs.jsonl"
            if conversation_log.exists():
                with open(conversation_log, "r") as f:
                    for line in f:
                        conv = json.loads(line)
                        # Filter for quality and privacy
                        if self.validate_conversation(conv):
                            new_conversations.append(conv)

            logger.info(f"✅ Collected {len(new_conversations)} new conversations")
            return new_conversations

        except Exception as e:
            logger.error(f"❌ Error collecting conversations: {e}")
            return []

    def validate_conversation(self, conversation: Dict) -> bool:
        """Validate conversation data quality and privacy compliance"""
        # Check required fields
        required = ["user_query", "model_response", "user_satisfied", "timestamp"]
        if not all(field in conversation for field in required):
            return False

        # Check privacy (no PII)
        if self.contains_pii(conversation["user_query"]):
            return False

        # Check quality
        if (
            len(conversation["user_query"]) < 10
            or len(conversation["model_response"]) < 20
        ):
            return False

        return True

    def contains_pii(self, text: str) -> bool:
        """Basic PII detection (in production, use more sophisticated methods)"""
        import re

        # Check for common PII patterns
        patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def check_model_performance(self) -> Dict:
        """Check current model performance metrics"""
        logger.info("📊 Checking current model performance...")

        try:
            # Get current production model from registry
            response = requests.get(f"{MODEL_REGISTRY_URL}/api/v1/models/production")
            if response.status_code == 200:
                current_model = response.json()

                # Get recent performance metrics
                metrics = {
                    "accuracy": 0.93,  # Would come from monitoring
                    "response_time_ms": 125,
                    "user_satisfaction": 0.87,
                    "crisis_detection_rate": 0.98,
                    "last_updated": current_model.get("updated_at", "unknown"),
                }

                logger.info(f"✅ Current model performance: {metrics}")
                return metrics
            else:
                logger.warning("⚠️  Could not fetch current model metrics")
                return {}

        except Exception as e:
            logger.error(f"❌ Error checking performance: {e}")
            return {}

    def should_retrain(
        self, new_conversations: List[Dict], current_metrics: Dict
    ) -> bool:
        """Determine if retraining is needed"""
        logger.info("🤔 Evaluating retraining criteria...")

        reasons = []

        # Check conversation volume
        if len(new_conversations) >= self.config["min_new_conversations"]:
            reasons.append(
                f"Sufficient new data: {len(new_conversations)} conversations"
            )

        # Check performance degradation
        baseline_accuracy = 0.95  # Original model accuracy
        current_accuracy = current_metrics.get("accuracy", 1.0)
        if (
            baseline_accuracy - current_accuracy
            > self.config["performance_degradation_threshold"]
        ):
            reasons.append(
                f"Performance degradation: {current_accuracy:.1%} vs {baseline_accuracy:.1%}"
            )

        # Check time since last update
        # In production, parse last_updated timestamp
        days_since_update = 10  # Simulated
        if days_since_update >= self.config["retraining_interval_days"]:
            reasons.append(f"Time threshold reached: {days_since_update} days")

        if reasons:
            logger.info(f"✅ Retraining recommended. Reasons: {', '.join(reasons)}")
            return True
        else:
            logger.info("❌ Retraining not needed at this time")
            return False

    def prepare_training_data(self, new_conversations: List[Dict]) -> str:
        """Prepare combined training dataset"""
        logger.info("📝 Preparing training data...")

        # Load existing training data
        base_data_path = Path("data/combined_healthcare_training_data.jsonl")
        combined_data = []

        if base_data_path.exists():
            with open(base_data_path, "r") as f:
                for line in f:
                    combined_data.append(json.loads(line))

        # Add new conversations
        for conv in new_conversations:
            training_sample = {
                "conversations": [
                    {"from": "human", "value": conv["user_query"]},
                    {"from": "assistant", "value": conv["model_response"]},
                ],
                "category": conv.get("category", "general"),
                "metadata": {
                    "user_satisfied": conv["user_satisfied"],
                    "timestamp": conv["timestamp"],
                    "source": "production_feedback",
                },
            }
            combined_data.append(training_sample)

        # Save combined dataset
        output_path = self.data_dir / f"training_data_{self.pipeline_id}.jsonl"
        with open(output_path, "w") as f:
            for sample in combined_data:
                f.write(json.dumps(sample) + "\n")

        logger.info(f"✅ Prepared {len(combined_data)} total training samples")
        return str(output_path)

    def trigger_training_job(self, training_data_path: str) -> str:
        """Trigger model training via Pipeline Orchestrator"""
        logger.info("🚀 Triggering training job...")

        # Create experiment in tracking system
        experiment_data = {
            "name": f"Auto-retrain {self.pipeline_id}",
            "description": "Automated retraining with production feedback",
            "tags": ["auto-retrain", "production", "healthcare"],
            "parameters": {
                "training_data": training_data_path,
                "model_type": "healthcare_ai_advanced",
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.0001,
                "fine_tuning": True,
            },
        }

        try:
            # Register experiment
            response = requests.post(
                f"{EXPERIMENT_TRACKING_URL}/api/v1/experiments", json=experiment_data
            )

            if response.status_code in [200, 201]:
                experiment_id = response.json().get("id", "unknown")
                logger.info(f"✅ Created experiment: {experiment_id}")

                # In production, this would submit to Pipeline Orchestrator
                # For demo, simulate training
                self.simulate_training(experiment_id)

                return experiment_id
            else:
                logger.error(f"❌ Failed to create experiment: {response.text}")
                return None

        except Exception as e:
            logger.error(f"❌ Error triggering training: {e}")
            return None

    def simulate_training(self, experiment_id: str):
        """Simulate model training process"""
        logger.info("🔄 Training model (simulated)...")

        # Simulate training progress
        for epoch in range(1, 6):
            time.sleep(2)  # Simulate training time

            metrics = {
                "epoch": epoch,
                "train_loss": 0.5 / epoch,
                "val_loss": 0.6 / epoch,
                "train_accuracy": 0.90 + (0.02 * epoch),
                "val_accuracy": 0.88 + (0.02 * epoch),
            }

            logger.info(f"   Epoch {epoch}/5 - Accuracy: {metrics['val_accuracy']:.1%}")

            # Log metrics to experiment tracking
            try:
                requests.post(
                    f"{EXPERIMENT_TRACKING_URL}/api/v1/experiments/{experiment_id}/metrics",
                    json={"metrics": metrics},
                )
            except:
                pass

    def evaluate_new_model(self, experiment_id: str) -> Dict:
        """Evaluate newly trained model"""
        logger.info("🧪 Evaluating new model...")

        # Simulated evaluation results
        evaluation = {
            "accuracy": 0.97,  # 2% improvement
            "response_time_ms": 130,
            "user_satisfaction_predicted": 0.91,
            "crisis_detection_rate": 0.995,
            "test_set_size": 5000,
            "improvement_over_baseline": 0.02,
        }

        logger.info(f"✅ New model evaluation: {evaluation}")
        return evaluation

    def deploy_model(self, experiment_id: str, evaluation: Dict) -> bool:
        """Deploy new model if it meets criteria"""
        logger.info("🚢 Evaluating deployment criteria...")

        improvement = evaluation.get("improvement_over_baseline", 0)

        if improvement >= self.config["auto_deploy_threshold"]:
            logger.info(f"✅ Auto-deployment approved: {improvement:.1%} improvement")

            # Register new model version
            model_data = {
                "name": "healthcare-ai/advanced-engine",
                "version": f"2.1.{int(time.time())}",
                "experiment_id": experiment_id,
                "stage": "staging",
                "metrics": evaluation,
                "tags": ["auto-retrained", "production-feedback"],
                "description": f"Auto-retrained model with {improvement:.1%} improvement",
            }

            try:
                response = requests.post(
                    f"{MODEL_REGISTRY_URL}/api/v1/models", json=model_data
                )

                if response.status_code in [200, 201]:
                    model_id = response.json().get("id", "unknown")
                    logger.info(f"✅ Registered new model: {model_id}")

                    # Promote to production (with safety checks in real deployment)
                    self.promote_to_production(model_id)
                    return True

            except Exception as e:
                logger.error(f"❌ Deployment failed: {e}")

        else:
            logger.info(
                f"❌ Improvement {improvement:.1%} below threshold {self.config['auto_deploy_threshold']:.1%}"
            )

        return False

    def promote_to_production(self, model_id: str):
        """Promote model to production with safety checks"""
        logger.info("🎯 Promoting model to production...")

        # In production, this would:
        # 1. Run final safety checks
        # 2. Create canary deployment
        # 3. Monitor for issues
        # 4. Gradually increase traffic
        # 5. Full promotion if successful

        logger.info(f"✅ Model {model_id} promoted to production!")

    def run_pipeline(self):
        """Execute the complete auto-retraining pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 AUTO-RETRAINING PIPELINE: {self.pipeline_id}")
        logger.info(f"{'='*60}\n")

        # Step 1: Collect new conversations
        new_conversations = self.collect_new_conversations()

        # Step 2: Check current model performance
        current_metrics = self.check_model_performance()

        # Step 3: Decide if retraining is needed
        if not self.should_retrain(new_conversations, current_metrics):
            logger.info("✅ Pipeline complete - no retraining needed")
            return

        # Step 4: Prepare training data
        training_data_path = self.prepare_training_data(new_conversations)

        # Step 5: Trigger training job
        experiment_id = self.trigger_training_job(training_data_path)
        if not experiment_id:
            logger.error("❌ Pipeline failed - could not start training")
            return

        # Step 6: Evaluate new model
        evaluation = self.evaluate_new_model(experiment_id)

        # Step 7: Deploy if criteria met
        if self.deploy_model(experiment_id, evaluation):
            logger.info("✅ Pipeline complete - new model deployed!")
        else:
            logger.info("✅ Pipeline complete - model not deployed")


def main():
    """Run auto-retraining pipeline"""
    pipeline = AutoRetrainingPipeline()

    # For demo, create some sample conversation data
    sample_conversations = []
    for i in range(1500):
        sample_conversations.append(
            {
                "user_query": f"I need help with medication management",
                "model_response": "I can help you manage medications effectively...",
                "user_satisfied": True if i % 10 != 0 else False,
                "timestamp": datetime.now().isoformat(),
                "category": "medication",
            }
        )

    # Save sample data
    with open(pipeline.data_dir / "conversation_logs.jsonl", "w") as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv) + "\n")

    # Run pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
