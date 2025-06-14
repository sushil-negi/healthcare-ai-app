#!/usr/bin/env python3
"""
Combine All Healthcare Datasets for Training
Merges original, production, real authority, and specialized datasets
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} conversations from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []


def combine_all_datasets():
    """Combine all healthcare datasets into comprehensive training set"""

    logger.info("🔄 Starting dataset combination process...")

    # Define all dataset paths
    datasets = {
        "original_healthcare": "/Users/snegi/Documents/github/mlops-project/data/healthcare_training_data.json",
        "production_healthcare": "/Users/snegi/Documents/github/mlops-project/data/production_healthcare_data.json",
        "real_healthcare": "/Users/snegi/Documents/github/mlops-project/data/real_healthcare_data.json",
        "adl_specialized": "/Users/snegi/Documents/github/mlops-project/data/adl_healthcare_data.json",
        "senior_care_specialized": "/Users/snegi/Documents/github/mlops-project/data/senior_care_healthcare_data.json",
        "mental_health_specialized": "/Users/snegi/Documents/github/mlops-project/data/mental_health_healthcare_data.json",
        "respite_care_specialized": "/Users/snegi/Documents/github/mlops-project/data/respite_care_healthcare_data.json",
        "disabilities_specialized": "/Users/snegi/Documents/github/mlops-project/data/disabilities_healthcare_data.json",
    }

    # Load all datasets
    all_conversations = []
    dataset_stats = {}

    for dataset_name, file_path in datasets.items():
        conversations = load_dataset(file_path)
        dataset_stats[dataset_name] = len(conversations)
        all_conversations.extend(conversations)

    # Generate combined statistics
    total_conversations = len(all_conversations)

    combined_stats = {
        "total_conversations": total_conversations,
        "combination_timestamp": datetime.now().isoformat(),
        "source_datasets": dataset_stats,
        "dataset_distribution": {
            "original_healthcare": dataset_stats.get("original_healthcare", 0),
            "production_healthcare": dataset_stats.get("production_healthcare", 0),
            "real_healthcare": dataset_stats.get("real_healthcare", 0),
            "specialized_adl": dataset_stats.get("adl_specialized", 0),
            "specialized_senior_care": dataset_stats.get("senior_care_specialized", 0),
            "specialized_mental_health": dataset_stats.get(
                "mental_health_specialized", 0
            ),
            "specialized_respite_care": dataset_stats.get(
                "respite_care_specialized", 0
            ),
            "specialized_disabilities": dataset_stats.get(
                "disabilities_specialized", 0
            ),
        },
        "training_ready": True,
        "version": "4.0.0",
    }

    # Save combined dataset
    combined_json_path = "/Users/snegi/Documents/github/mlops-project/data/combined_healthcare_training_data.json"
    combined_jsonl_path = "/Users/snegi/Documents/github/mlops-project/data/combined_healthcare_training_data.jsonl"
    combined_stats_path = "/Users/snegi/Documents/github/mlops-project/data/combined_healthcare_training_stats.json"

    # Save as JSON
    with open(combined_json_path, "w") as f:
        json.dump(all_conversations, f, indent=2)

    # Save as JSONL for training
    with open(combined_jsonl_path, "w") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv) + "\n")

    # Save statistics
    with open(combined_stats_path, "w") as f:
        json.dump(combined_stats, f, indent=2)

    logger.info(f"✅ Combined dataset creation complete!")
    logger.info(f"📊 Total conversations: {total_conversations}")
    logger.info(f"📁 Files created:")
    logger.info(f"  JSON: {combined_json_path}")
    logger.info(f"  JSONL: {combined_jsonl_path}")
    logger.info(f"  Stats: {combined_stats_path}")

    logger.info(f"\n📈 Dataset Breakdown:")
    for dataset_name, count in dataset_stats.items():
        percentage = (
            (count / total_conversations) * 100 if total_conversations > 0 else 0
        )
        logger.info(f"  {dataset_name}: {count:,} conversations ({percentage:.1f}%)")

    return combined_stats


if __name__ == "__main__":
    combine_all_datasets()
