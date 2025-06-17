#!/usr/bin/env python3
"""
Automated Model Promotion Manager
Handles safe promotion of models from staging to production
"""

import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import requests

# MLOps Platform Services
MODEL_REGISTRY_URL = "http://localhost:8000"
AB_TESTING_URL = "http://localhost:8090"
MONITORING_URL = "http://localhost:9090"  # Prometheus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromotionStage(Enum):
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    GRADUAL_ROLLOUT = "gradual_rollout"
    INSTANT = "instant"


class ModelPromotionManager:
    def __init__(self):
        self.promotion_criteria = {
            "min_evaluation_period_hours": 24,
            "min_requests_served": 1000,
            "max_error_rate": 0.05,
            "min_accuracy_improvement": 0.01,  # 1% improvement
            "max_latency_increase_ms": 50,
            "crisis_detection_threshold": 0.98,  # Must maintain 98%+
            "user_satisfaction_threshold": 0.85,
        }

        self.rollout_stages = {
            PromotionStage.CANARY: [1, 5, 10],  # Traffic percentages
            PromotionStage.GRADUAL_ROLLOUT: [10, 25, 50, 75, 100],
            PromotionStage.BLUE_GREEN: [0, 100],
            PromotionStage.INSTANT: [100],
        }

    def evaluate_candidate_model(self, model_id: str, current_model_id: str) -> Dict:
        """Evaluate if candidate model is ready for promotion"""
        logger.info(f"🔍 Evaluating candidate model {model_id} for promotion...")

        evaluation = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "ready_for_promotion": False,
            "criteria_results": {},
            "risk_score": 0.0,
            "recommended_strategy": None,
        }

        # Get model metadata from registry
        try:
            candidate = self.get_model_info(model_id)
            current = self.get_model_info(current_model_id)

            if not candidate or not current:
                evaluation["criteria_results"][
                    "model_info"
                ] = "Failed to retrieve model information"
                return evaluation

        except Exception as e:
            logger.error(f"Error retrieving model info: {e}")
            return evaluation

        # 1. Check evaluation period
        model_age_hours = self.calculate_model_age_hours(candidate)
        evaluation["criteria_results"]["evaluation_period"] = {
            "hours_in_staging": model_age_hours,
            "meets_criteria": model_age_hours
            >= self.promotion_criteria["min_evaluation_period_hours"],
        }

        # 2. Check request volume
        requests_served = self.get_requests_served(model_id)
        evaluation["criteria_results"]["request_volume"] = {
            "requests_served": requests_served,
            "meets_criteria": requests_served
            >= self.promotion_criteria["min_requests_served"],
        }

        # 3. Check error rate
        error_rate = self.get_error_rate(model_id)
        evaluation["criteria_results"]["error_rate"] = {
            "current_rate": error_rate,
            "meets_criteria": error_rate <= self.promotion_criteria["max_error_rate"],
        }

        # 4. Check performance improvement
        perf_comparison = self.compare_model_performance(candidate, current)
        evaluation["criteria_results"]["performance"] = perf_comparison

        # 5. Check critical safety metrics
        safety_check = self.check_safety_metrics(model_id)
        evaluation["criteria_results"]["safety"] = safety_check

        # Calculate overall readiness
        all_criteria_met = all(
            criteria.get("meets_criteria", False)
            for criteria in evaluation["criteria_results"].values()
            if isinstance(criteria, dict) and "meets_criteria" in criteria
        )

        evaluation["ready_for_promotion"] = all_criteria_met

        # Calculate risk score and recommend strategy
        evaluation["risk_score"] = self.calculate_risk_score(evaluation)
        evaluation["recommended_strategy"] = self.recommend_promotion_strategy(
            evaluation
        )

        logger.info(
            f"✅ Evaluation complete: Ready={evaluation['ready_for_promotion']}, "
            f"Risk={evaluation['risk_score']:.2f}, Strategy={evaluation['recommended_strategy']}"
        )

        return evaluation

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Retrieve model information from registry"""
        try:
            response = requests.get(f"{MODEL_REGISTRY_URL}/api/v1/models/{model_id}")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

    def calculate_model_age_hours(self, model: Dict) -> float:
        """Calculate how long model has been in staging"""
        # In production, parse created_at timestamp
        return 48.0  # Simulated 48 hours

    def get_requests_served(self, model_id: str) -> int:
        """Get number of requests served by model"""
        # In production, query Prometheus/monitoring system
        return 5000  # Simulated

    def get_error_rate(self, model_id: str) -> float:
        """Get model error rate from monitoring"""
        # In production, query metrics
        return 0.03  # Simulated 3% error rate

    def compare_model_performance(self, candidate: Dict, current: Dict) -> Dict:
        """Compare performance between candidate and current model"""
        # Simulated comparison
        candidate_metrics = candidate.get("metrics", {})
        current_metrics = current.get("metrics", {})

        accuracy_improvement = candidate_metrics.get(
            "accuracy", 0
        ) - current_metrics.get("accuracy", 0)
        latency_increase = candidate_metrics.get("latency_ms", 0) - current_metrics.get(
            "latency_ms", 0
        )

        return {
            "accuracy_improvement": accuracy_improvement,
            "latency_increase_ms": latency_increase,
            "meets_criteria": (
                accuracy_improvement
                >= self.promotion_criteria["min_accuracy_improvement"]
                and latency_increase
                <= self.promotion_criteria["max_latency_increase_ms"]
            ),
        }

    def check_safety_metrics(self, model_id: str) -> Dict:
        """Check critical safety metrics for healthcare"""
        # Simulated safety check
        return {
            "crisis_detection_rate": 0.99,
            "user_satisfaction": 0.88,
            "medical_accuracy": 0.96,
            "meets_criteria": True,
        }

    def calculate_risk_score(self, evaluation: Dict) -> float:
        """Calculate promotion risk score (0-1, lower is better)"""
        risk_factors = []

        # High risk if safety criteria barely met
        safety = evaluation["criteria_results"].get("safety", {})
        if safety.get("crisis_detection_rate", 0) < 0.99:
            risk_factors.append(0.5)

        # Medium risk for performance regression
        perf = evaluation["criteria_results"].get("performance", {})
        if perf.get("latency_increase_ms", 0) > 25:
            risk_factors.append(0.3)

        # Low risk for minimal testing
        volume = evaluation["criteria_results"].get("request_volume", {})
        if volume.get("requests_served", 0) < 5000:
            risk_factors.append(0.2)

        return min(sum(risk_factors), 1.0)

    def recommend_promotion_strategy(self, evaluation: Dict) -> PromotionStage:
        """Recommend promotion strategy based on risk"""
        risk_score = evaluation["risk_score"]

        if risk_score < 0.2:
            return PromotionStage.BLUE_GREEN
        elif risk_score < 0.5:
            return PromotionStage.GRADUAL_ROLLOUT
        else:
            return PromotionStage.CANARY

    def execute_promotion(self, model_id: str, strategy: PromotionStage) -> bool:
        """Execute model promotion with specified strategy"""
        logger.info(f"🚀 Executing {strategy.value} promotion for model {model_id}")

        stages = self.rollout_stages[strategy]

        for i, traffic_percentage in enumerate(stages):
            logger.info(
                f"   Stage {i+1}/{len(stages)}: Routing {traffic_percentage}% traffic"
            )

            # Update traffic routing
            if not self.update_traffic_routing(model_id, traffic_percentage):
                logger.error(f"❌ Failed to update traffic routing")
                self.rollback_promotion(model_id)
                return False

            # Monitor for issues (except for instant promotion)
            if strategy != PromotionStage.INSTANT and traffic_percentage < 100:
                monitoring_duration = 300 if strategy == PromotionStage.CANARY else 60
                if not self.monitor_deployment(model_id, monitoring_duration):
                    logger.error(f"❌ Issues detected during monitoring")
                    self.rollback_promotion(model_id)
                    return False

        # Final promotion to production
        if self.finalize_promotion(model_id):
            logger.info(f"✅ Model {model_id} successfully promoted to production!")
            return True
        else:
            logger.error(f"❌ Failed to finalize promotion")
            return False

    def update_traffic_routing(self, model_id: str, percentage: float) -> bool:
        """Update traffic routing configuration"""
        try:
            # In production, update load balancer/service mesh
            config = {
                "model_id": model_id,
                "traffic_percentage": percentage,
                "timestamp": datetime.now().isoformat(),
            }

            # Simulate API call to AB testing service
            logger.info(f"   Updated routing: {percentage}% → {model_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating traffic: {e}")
            return False

    def monitor_deployment(self, model_id: str, duration_seconds: int) -> bool:
        """Monitor deployment for issues"""
        logger.info(f"   Monitoring for {duration_seconds}s...")

        # Simulate monitoring
        time.sleep(2)  # In production, actually wait and check metrics

        # Check key metrics
        metrics = {
            "error_rate": 0.02,
            "response_time_p99": 150,
            "crisis_detection_rate": 0.99,
        }

        # Validate against thresholds
        if metrics["error_rate"] > 0.10:  # 10% error rate threshold
            logger.error(f"   ⚠️ High error rate detected: {metrics['error_rate']:.1%}")
            return False

        if metrics["crisis_detection_rate"] < 0.98:
            logger.error(
                f"   ⚠️ Crisis detection degraded: {metrics['crisis_detection_rate']:.1%}"
            )
            return False

        logger.info(f"   ✅ Monitoring passed")
        return True

    def rollback_promotion(self, model_id: str):
        """Rollback failed promotion"""
        logger.warning(f"🔄 Rolling back promotion for model {model_id}")

        # Restore 100% traffic to previous model
        self.update_traffic_routing("current-production-model", 100)

        # Update model status in registry
        try:
            requests.patch(
                f"{MODEL_REGISTRY_URL}/api/v1/models/{model_id}",
                json={"stage": "failed"},
            )
        except:
            pass

        logger.info("✅ Rollback completed")

    def finalize_promotion(self, model_id: str) -> bool:
        """Finalize model promotion"""
        try:
            # Update model registry
            requests.patch(
                f"{MODEL_REGISTRY_URL}/api/v1/models/{model_id}",
                json={"stage": "production"},
            )

            # Archive previous model
            # Update monitoring dashboards
            # Send notifications

            return True

        except Exception as e:
            logger.error(f"Error finalizing promotion: {e}")
            return False

    def create_promotion_report(self, evaluation: Dict, success: bool) -> Dict:
        """Create detailed promotion report"""
        return {
            "promotion_id": f"promo-{int(time.time())}",
            "model_id": evaluation["model_id"],
            "timestamp": datetime.now().isoformat(),
            "evaluation_results": evaluation,
            "promotion_success": success,
            "strategy_used": (
                evaluation["recommended_strategy"].value
                if evaluation["recommended_strategy"]
                else None
            ),
            "risk_score": evaluation["risk_score"],
            "next_steps": (
                [
                    "Monitor production metrics for 24 hours",
                    "Prepare rollback plan if issues arise",
                    "Schedule post-promotion review",
                ]
                if success
                else [
                    "Review failure reasons",
                    "Address identified issues",
                    "Re-evaluate after fixes",
                ]
            ),
        }


def demonstrate_model_promotion():
    """Demonstrate automated model promotion"""
    manager = ModelPromotionManager()

    print("\n" + "=" * 60)
    print("🚀 AUTOMATED MODEL PROMOTION DEMO")
    print("=" * 60 + "\n")

    # Simulate model evaluation
    candidate_model = "healthcare-ai-v2.1"
    current_model = "healthcare-ai-v2.0"

    print(f"1️⃣ Evaluating candidate model: {candidate_model}")
    evaluation = manager.evaluate_candidate_model(candidate_model, current_model)

    print(f"\n📊 Evaluation Results:")
    print(f"   Ready for promotion: {evaluation['ready_for_promotion']}")
    print(f"   Risk score: {evaluation['risk_score']:.2f}")
    print(f"   Recommended strategy: {evaluation['recommended_strategy'].value}")

    if evaluation["ready_for_promotion"]:
        print(f"\n2️⃣ Executing {evaluation['recommended_strategy'].value} promotion...")
        success = manager.execute_promotion(
            candidate_model, evaluation["recommended_strategy"]
        )

        # Generate report
        report = manager.create_promotion_report(evaluation, success)
        print(f"\n📋 Promotion Report:")
        print(json.dumps(report, indent=2))
    else:
        print("\n❌ Model not ready for promotion")
        print("Issues to address:")
        for criteria, result in evaluation["criteria_results"].items():
            if isinstance(result, dict) and not result.get("meets_criteria", True):
                print(f"   - {criteria}: {result}")


if __name__ == "__main__":
    demonstrate_model_promotion()
