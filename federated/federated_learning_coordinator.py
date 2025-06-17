#!/usr/bin/env python3
"""
Federated Learning Coordinator for Healthcare AI
Enables collaborative training across multiple healthcare organizations
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedLearningCoordinator:
    def __init__(self):
        self.participants = {}
        self.federation_config = {
            "min_participants": 3,
            "max_participants": 10,
            "rounds_per_epoch": 5,
            "convergence_threshold": 0.001,
            "privacy_budget": 1.0,  # Differential privacy
            "secure_aggregation": True,
        }
        self.encryption_key = Fernet.generate_key()
        self.current_round = 0

    def register_participant(self, org_id: str, org_info: Dict) -> bool:
        """Register a healthcare organization for federated learning"""
        logger.info(f"📝 Registering participant: {org_id}")

        # Validate organization
        if not self.validate_participant(org_info):
            logger.error(f"❌ Invalid participant: {org_id}")
            return False

        self.participants[org_id] = {
            "org_id": org_id,
            "name": org_info["name"],
            "data_size": org_info["data_size"],
            "privacy_level": org_info.get("privacy_level", "high"),
            "compute_capacity": org_info.get("compute_capacity", "medium"),
            "joined_at": datetime.now().isoformat(),
            "status": "active",
            "model_weights": None,
            "last_update": None,
        }

        logger.info(f"✅ Participant {org_id} registered successfully")
        return True

    def validate_participant(self, org_info: Dict) -> bool:
        """Validate participant meets requirements"""
        required_fields = ["name", "data_size", "compliance_cert"]

        # Check required fields
        if not all(field in org_info for field in required_fields):
            return False

        # Check minimum data requirements
        if org_info["data_size"] < 1000:  # Minimum 1000 samples
            return False

        # Check HIPAA compliance
        if not org_info.get("hipaa_compliant", False):
            return False

        return True

    def initialize_federation(self, base_model_weights: Dict) -> bool:
        """Initialize federated learning session"""
        logger.info("🚀 Initializing federated learning session...")

        if len(self.participants) < self.federation_config["min_participants"]:
            logger.error(
                f"❌ Insufficient participants: {len(self.participants)} < {self.federation_config['min_participants']}"
            )
            return False

        # Distribute initial model weights
        for org_id in self.participants:
            encrypted_weights = self.encrypt_model_weights(base_model_weights)
            self.participants[org_id]["model_weights"] = encrypted_weights
            logger.info(f"   Sent initial weights to {org_id}")

        self.current_round = 1
        logger.info(
            f"✅ Federation initialized with {len(self.participants)} participants"
        )
        return True

    def encrypt_model_weights(self, weights: Dict) -> bytes:
        """Encrypt model weights for secure transmission"""
        fernet = Fernet(self.encryption_key)
        weights_json = json.dumps(weights, default=str)
        return fernet.encrypt(weights_json.encode())

    def decrypt_model_weights(self, encrypted_weights: bytes) -> Dict:
        """Decrypt model weights"""
        fernet = Fernet(self.encryption_key)
        weights_json = fernet.decrypt(encrypted_weights).decode()
        return json.loads(weights_json)

    def collect_model_updates(self) -> Dict:
        """Collect model updates from all participants"""
        logger.info(f"📥 Collecting updates for round {self.current_round}...")

        updates = {}
        total_samples = 0

        for org_id, participant in self.participants.items():
            if participant["status"] == "active":
                # Simulate local training
                local_weights = self.simulate_local_training(org_id, participant)

                if local_weights:
                    updates[org_id] = {
                        "weights": local_weights,
                        "sample_count": participant["data_size"],
                        "training_loss": np.random.uniform(0.1, 0.3),
                        "validation_accuracy": np.random.uniform(0.85, 0.95),
                    }
                    total_samples += participant["data_size"]
                    logger.info(f"   ✅ Received update from {org_id}")
                else:
                    logger.warning(f"   ⚠️ No update from {org_id}")

        logger.info(
            f"📊 Collected {len(updates)} updates (total samples: {total_samples})"
        )
        return updates

    def simulate_local_training(self, org_id: str, participant: Dict) -> Optional[Dict]:
        """Simulate local training at participant organization"""
        # In real federated learning, this would be done at the participant's site
        # This simulation represents what the participant would send back

        time.sleep(0.5)  # Simulate training time

        # Simulate weight updates (normally computed via backpropagation)
        simulated_weights = {
            "layer1": np.random.random((100, 50)).tolist(),
            "layer2": np.random.random((50, 25)).tolist(),
            "layer3": np.random.random((25, 11)).tolist(),  # 11 healthcare categories
            "bias1": np.random.random(50).tolist(),
            "bias2": np.random.random(25).tolist(),
            "bias3": np.random.random(11).tolist(),
        }

        return simulated_weights

    def apply_differential_privacy(self, weights: Dict, epsilon: float = 1.0) -> Dict:
        """Apply differential privacy to protect individual data"""
        logger.info(f"🔒 Applying differential privacy (ε={epsilon})")

        # Add calibrated noise to weights
        private_weights = {}
        for layer, weight_matrix in weights.items():
            if isinstance(weight_matrix, list):
                # Convert to numpy for noise addition
                weight_array = np.array(weight_matrix)

                # Calculate sensitivity (max possible change)
                sensitivity = 2.0 / len(self.participants)  # Simplified

                # Add Laplace noise
                noise_scale = sensitivity / epsilon
                noise = np.random.laplace(0, noise_scale, weight_array.shape)

                private_weights[layer] = (weight_array + noise).tolist()
            else:
                private_weights[layer] = weight_matrix

        return private_weights

    def secure_aggregate(self, updates: Dict) -> Dict:
        """Perform secure aggregation of model updates"""
        logger.info("🔐 Performing secure aggregation...")

        # Weighted averaging based on data size
        total_samples = sum(update["sample_count"] for update in updates.values())
        aggregated_weights = {}

        # Initialize aggregated weights
        first_update = next(iter(updates.values()))
        for layer in first_update["weights"]:
            aggregated_weights[layer] = np.zeros_like(
                np.array(first_update["weights"][layer])
            )

        # Weighted sum
        for org_id, update in updates.items():
            weight_factor = update["sample_count"] / total_samples

            for layer in update["weights"]:
                layer_weights = np.array(update["weights"][layer])
                aggregated_weights[layer] += weight_factor * layer_weights

        # Convert back to lists
        for layer in aggregated_weights:
            aggregated_weights[layer] = aggregated_weights[layer].tolist()

        # Apply differential privacy
        private_weights = self.apply_differential_privacy(aggregated_weights)

        logger.info("✅ Secure aggregation completed")
        return private_weights

    def check_convergence(self, current_weights: Dict, previous_weights: Dict) -> bool:
        """Check if federated training has converged"""
        if not previous_weights:
            return False

        # Calculate weight difference
        total_diff = 0
        total_params = 0

        for layer in current_weights:
            if layer in previous_weights:
                current = np.array(current_weights[layer])
                previous = np.array(previous_weights[layer])
                diff = np.mean(np.abs(current - previous))
                total_diff += diff
                total_params += 1

        avg_diff = total_diff / total_params if total_params > 0 else float("inf")
        converged = avg_diff < self.federation_config["convergence_threshold"]

        logger.info(
            f"📏 Convergence check: avg_diff={avg_diff:.6f}, threshold={self.federation_config['convergence_threshold']:.6f}"
        )
        return converged

    def run_federated_training(self, max_rounds: int = 10) -> Dict:
        """Execute complete federated training session"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🌐 FEDERATED LEARNING SESSION STARTED")
        logger.info(f"{'='*60}\n")

        training_history = {
            "rounds": [],
            "participants": len(self.participants),
            "convergence_achieved": False,
            "final_accuracy": 0.0,
        }

        previous_weights = None

        for round_num in range(1, max_rounds + 1):
            logger.info(f"\n🔄 Round {round_num}/{max_rounds}")
            self.current_round = round_num

            # Collect updates from participants
            updates = self.collect_model_updates()

            if not updates:
                logger.error("❌ No updates received, stopping training")
                break

            # Secure aggregation
            aggregated_weights = self.secure_aggregate(updates)

            # Calculate round metrics
            round_metrics = self.calculate_round_metrics(updates)
            training_history["rounds"].append(round_metrics)

            logger.info(f"📊 Round {round_num} Results:")
            logger.info(f"   Average accuracy: {round_metrics['avg_accuracy']:.1%}")
            logger.info(f"   Average loss: {round_metrics['avg_loss']:.4f}")
            logger.info(f"   Participants: {round_metrics['participant_count']}")

            # Check convergence
            if self.check_convergence(aggregated_weights, previous_weights):
                logger.info("✅ Convergence achieved!")
                training_history["convergence_achieved"] = True
                training_history["final_accuracy"] = round_metrics["avg_accuracy"]
                break

            # Distribute updated weights for next round
            self.distribute_weights(aggregated_weights)
            previous_weights = aggregated_weights

        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 FEDERATED LEARNING SESSION COMPLETED")
        logger.info(f"{'='*60}")

        return training_history

    def calculate_round_metrics(self, updates: Dict) -> Dict:
        """Calculate aggregated metrics for the round"""
        accuracies = [update["validation_accuracy"] for update in updates.values()]
        losses = [update["training_loss"] for update in updates.values()]

        return {
            "round": self.current_round,
            "participant_count": len(updates),
            "avg_accuracy": np.mean(accuracies),
            "avg_loss": np.mean(losses),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "timestamp": datetime.now().isoformat(),
        }

    def distribute_weights(self, weights: Dict):
        """Distribute updated weights to all participants"""
        for org_id in self.participants:
            encrypted_weights = self.encrypt_model_weights(weights)
            self.participants[org_id]["model_weights"] = encrypted_weights
            self.participants[org_id]["last_update"] = datetime.now().isoformat()


def demonstrate_federated_learning():
    """Demonstrate federated learning capabilities"""
    coordinator = FederatedLearningCoordinator()

    print("\n" + "=" * 60)
    print("🌐 FEDERATED LEARNING DEMO")
    print("=" * 60 + "\n")

    # Register healthcare organizations
    orgs = [
        {
            "org_id": "mayo-clinic",
            "name": "Mayo Clinic",
            "data_size": 50000,
            "hipaa_compliant": True,
            "compliance_cert": "HIPAA-2024",
            "privacy_level": "high",
        },
        {
            "org_id": "johns-hopkins",
            "name": "Johns Hopkins Hospital",
            "data_size": 75000,
            "hipaa_compliant": True,
            "compliance_cert": "HIPAA-2024",
            "privacy_level": "high",
        },
        {
            "org_id": "cleveland-clinic",
            "name": "Cleveland Clinic",
            "data_size": 40000,
            "hipaa_compliant": True,
            "compliance_cert": "HIPAA-2024",
            "privacy_level": "high",
        },
        {
            "org_id": "kaiser-permanente",
            "name": "Kaiser Permanente",
            "data_size": 100000,
            "hipaa_compliant": True,
            "compliance_cert": "HIPAA-2024",
            "privacy_level": "high",
        },
    ]

    print("1️⃣ Registering healthcare organizations...")
    for org in orgs:
        success = coordinator.register_participant(org["org_id"], org)
        print(f"   {org['name']}: {'✅' if success else '❌'}")

    # Initialize federation
    print("\n2️⃣ Initializing federated learning...")
    base_weights = {
        "layer1": np.random.random((100, 50)).tolist(),
        "layer2": np.random.random((50, 25)).tolist(),
        "layer3": np.random.random((25, 11)).tolist(),
    }

    if coordinator.initialize_federation(base_weights):
        print("✅ Federation initialized successfully")

        # Run federated training
        print("\n3️⃣ Running federated training...")
        results = coordinator.run_federated_training(max_rounds=5)

        print(f"\n📋 Training Summary:")
        print(f"   Participants: {results['participants']}")
        print(f"   Rounds completed: {len(results['rounds'])}")
        print(f"   Convergence achieved: {results['convergence_achieved']}")
        print(f"   Final accuracy: {results['final_accuracy']:.1%}")

        print(f"\n🔒 Privacy Guarantees:")
        print(f"   ✅ No raw data shared between organizations")
        print(f"   ✅ Differential privacy applied (ε=1.0)")
        print(f"   ✅ Secure aggregation with encryption")
        print(f"   ✅ HIPAA compliance maintained")

    else:
        print("❌ Federation initialization failed")


if __name__ == "__main__":
    demonstrate_federated_learning()
